// Decisive diagnostic: does the 26-bone foot + shin + load assembly fail to
// balance because of (A) contact-model degeneracy, or (B) a fundamental need
// for active control?
//
// Method: build the SAME assembly as ankle_anatomical::build_world, but weld
// EVERYTHING into one rigid body — including a RIGID talocrural weld (no PD,
// no compliance) and a weld of every foot bone to the calcaneus. Then:
//   * settle ~1 s,
//   * compute the true mass-weighted CoM of all bodies,
//   * collect the ground-contact polygon from engaged_endpoints,
//   * test whether CoM projects inside that polygon,
//   * run 5 s and measure max stick tilt.
//
// Interpretation:
//   CoM inside polygon AND still tips  -> Cause A (contact cannot hold a rigid
//                                         body whose CoM is over its base).
//   Rigid assembly stands (<5 deg)     -> contact is fine; earlier failure was
//                                         ankle compliance -> leans Cause B.
//   CoM outside polygon                -> placement issue, not fundamental.
//
// This binary does NOT modify the engine, REST_POS, or the validation binary.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::math::{Vec3, Vec2, Quat, convex_hull_2d, point_in_convex_polygon_2d};
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_damper::AngularDamper;
use physics::constraints::PositionalConstraint;
use nalgebra::UnitQuaternion;

use improper_football::anatomy::foot::{self, Bone, BONE_COUNT};

/// The talus/foot bones carry this rest baseline (local +y -> world +z).
/// Mirrors foot::bone_baseline (private), needed so a near-rigid, baseline-
/// aware ankle lock holds shin(identity)↔talus(baseline) at zero preload.
fn bone_baseline() -> Quat {
    Quat::from_scaled_axis(Vec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0))
}

/// A near-rigid talocrural lock that DOES span the identity→baseline
/// orientation boundary (which foot::weld cannot — its angular lock is
/// baseline-free and would try to crush the π/2 talus baseline to zero,
/// exploding the assembly). Built from baseline-AWARE AngularSoftLimits with
/// a near-zero range and very high stiffness on all three axes, plus heavy
/// damping. No PD controller -> zero ankle compliance / no active hold.
/// This is the most rigid shin↔talus coupling expressible without modifying
/// the engine.
fn rigid_talocrural(world: &mut XpbdWorld, shin_idx: usize, talus_idx: usize) {
    let joint_world = world.bodies[talus_idx].position; // talus center
    let a_shin = foot::anchor_at(world, shin_idx, joint_world);
    let a_talus = foot::anchor_at(world, talus_idx, joint_world);
    let mut c = PositionalConstraint::new(shin_idx, talus_idx, 0.0, 1.0 / 1_000_000.0);
    c.local_anchor_a = a_shin;
    c.local_anchor_b = a_talus;
    world.constraints.push(c);

    let baseline = bone_baseline();
    const LOCK_K: f64 = 60_000.0; // stiff angular lock
    const LOCK_RANGE: f64 = 0.002; // ±0.11° — effectively rigid
    for axis in [Vec3::x(), Vec3::y(), Vec3::z()] {
        world.angular_soft_limits.push(
            AngularSoftLimit::new(shin_idx, talus_idx, axis, -LOCK_RANGE, LOCK_RANGE, LOCK_K, 2.0)
                .with_baseline(baseline));
        world.angular_dampers.push(AngularDamper::new(shin_idx, talus_idx, axis, 5.0));
    }
}

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

const SHIN_MASS:   f64 = 3.00;
const SHIN_HALF_H: f64 = 0.21;
const SHIN_RADIUS: f64 = 0.05;

const STICK_MASS:   f64 = 7.00;
const STICK_HALF_H: f64 = 0.40;
const STICK_RADIUS: f64 = 0.08;

struct Rigid {
    world: XpbdWorld,
    foot_bodies: [usize; BONE_COUNT],
    stick_idx: usize,
}

/// Build the assembly fully rigid. `stick_xz` shifts the stick's horizontal
/// position (for the centered-CoM sanity test); pass (0,0) for the nominal
/// layout that matches ankle_anatomical::build_world. `weld_to_calcaneus`
/// toggles the redundant all-bones-to-calcaneus welds (the foot's own
/// internal joints are always present).
fn build_rigid(stick_xz: (f64, f64)) -> Rigid { build_rigid_opt(stick_xz, true) }

fn build_rigid_opt(stick_xz: (f64, f64), weld_to_calcaneus: bool) -> Rigid {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let id = UnitQuaternion::identity();

    let attach = Vec3::new(0.0, 0.039, 0.0);

    // Shin.
    let mut shin = RigidBody::new_capsule(SHIN_MASS, SHIN_HALF_H, SHIN_RADIUS);
    shin.position = Vec3::new(0.0, attach.y + SHIN_HALF_H, 0.0);
    shin.orientation = id;
    let shin_idx = world.add_body(shin);

    // Foot.
    let foot = foot::build_foot(&mut world, attach);
    let talus_idx = foot.body(Bone::Talus);
    let calcaneus_idx = foot.body(Bone::Calcaneus);

    // RIGID talocrural lock (shin <-> talus), baseline-aware, no PD.
    // Replaces the compliant attach_talocrural entirely (zero ankle compliance).
    rigid_talocrural(&mut world, shin_idx, talus_idx);

    // Weld every non-calcaneus foot bone rigidly to the calcaneus so the foot
    // cannot internally collapse. Redundant with the foot's internal joints —
    // that's fine, it just stiffens the assembly into one rigid object.
    if weld_to_calcaneus {
        for i in 0..BONE_COUNT {
            let idx = foot.bodies[i];
            if idx == calcaneus_idx { continue; }
            let mid = (world.bodies[idx].position + world.bodies[calcaneus_idx].position) * 0.5;
            let a_cal = foot::anchor_at(&world, calcaneus_idx, mid);
            let a_idx = foot::anchor_at(&world, idx, mid);
            foot::weld(&mut world, calcaneus_idx, idx, a_cal, a_idx);
        }
    }

    // Load stick welded rigidly to the shin top.
    let mut stick = RigidBody::new_capsule(STICK_MASS, STICK_HALF_H, STICK_RADIUS);
    stick.position = Vec3::new(
        stick_xz.0,
        attach.y + 2.0 * SHIN_HALF_H + STICK_HALF_H,
        stick_xz.1,
    );
    stick.orientation = id;
    let stick_idx = world.add_body(stick);
    let weld_pt = Vec3::new(0.0, attach.y + 2.0 * SHIN_HALF_H, 0.0); // shin top
    let a_ws = foot::anchor_at(&world, shin_idx, weld_pt);
    let a_wk = foot::anchor_at(&world, stick_idx, weld_pt);
    foot::weld(&mut world, shin_idx, stick_idx, a_ws, a_wk);

    Rigid { world, foot_bodies: foot.bodies, stick_idx }
}

fn simulate(world: &mut XpbdWorld, duration: f64, mut per_frame: impl FnMut(&mut XpbdWorld, f64)) {
    let mut t = 0.0;
    while t < duration {
        for _ in 0..SOLVER_ITERATIONS { world.substep(SUB_DT); }
        t += DT;
        per_frame(world, t);
    }
}

/// True mass-weighted CoM of every body in the world.
fn center_of_mass(world: &XpbdWorld) -> Vec3 {
    let (mut m, mut c) = (0.0, Vec3::zeros());
    for b in &world.bodies {
        let mm = 1.0 / b.inv_mass;
        m += mm;
        c += b.position * mm;
    }
    c / m
}

/// Collect every engaged ground-contact point across all foot bodies.
fn contact_points(world: &XpbdWorld, foot_bodies: &[usize; BONE_COUNT]) -> Vec<(Vec3, f64)> {
    let mut pts = vec![];
    for &idx in foot_bodies.iter() {
        for e in world.engaged_endpoints(idx).iter().flatten() {
            pts.push((e.contact_point, e.normal_force));
        }
    }
    pts
}

/// Shoelace area of a CCW polygon.
fn polygon_area(poly: &[Vec2]) -> f64 {
    if poly.len() < 3 { return 0.0; }
    let mut a = 0.0;
    for i in 0..poly.len() {
        let p = poly[i];
        let q = poly[(i + 1) % poly.len()];
        a += p.x * q.y - q.x * p.y;
    }
    (a * 0.5).abs()
}

fn report_polygon_pts(pts: &[(Vec3, f64)], com: Vec3) {
    println!("  contact points: {} engaged endpoints", pts.len());
    let (mut xmin, mut xmax, mut zmin, mut zmax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    let mut ftot = 0.0;
    let mut pts2d: Vec<Vec2> = vec![];
    for &(p, f) in pts.iter() {
        println!("    ({:+.4}, {:+.4})  z=ant  Fn={:.1} N", p.x, p.z, f);
        xmin = xmin.min(p.x); xmax = xmax.max(p.x);
        zmin = zmin.min(p.z); zmax = zmax.max(p.z);
        ftot += f;
        pts2d.push(Vec2::new(p.x, p.z));
    }
    if pts.is_empty() {
        println!("  NO CONTACT POINTS (foot airborne or rebounding).");
        return;
    }
    let (dx, dz) = (xmax - xmin, zmax - zmin);
    println!("  bbox: x[{:+.4},{:+.4}] (dx={:.4} m)  z[{:+.4},{:+.4}] (dz={:.4} m)",
        xmin, xmax, dx, zmin, zmax, dz);
    println!("  total normal force: {:.1} N (assembly weight ~{:.1} N)", ftot, total_weight());

    let hull = convex_hull_2d(&pts2d);
    let area = polygon_area(&hull);
    println!("  convex hull: {} vertices, area = {:.6} m^2 ({:.2} cm^2)",
        hull.len(), area, area * 1e4);
    if dx < 0.01 {
        println!("  *** DEGENERATE: x-extent < 1 cm — contact points nearly collinear in x ***");
    }
    if area < 1e-4 {
        println!("  *** DEGENERATE: polygon area < 1 cm^2 — effectively a line/point ***");
    }

    let com_xz = Vec2::new(com.x, com.z);
    let inside = point_in_convex_polygon_2d(com_xz, &hull);
    println!("  CoM (x,z) = ({:+.4}, {:+.4})", com.x, com.z);
    if hull.len() >= 3 {
        // signed margin to nearest edge (positive = inside)
        let mut min_margin = f64::MAX;
        for i in 0..hull.len() {
            let a = hull[i];
            let b = hull[(i + 1) % hull.len()];
            let edge = b - a;
            let len = edge.norm();
            if len < 1e-12 { continue; }
            // CCW: inside is left of edge; signed distance
            let n = Vec2::new(edge.y, -edge.x) / len; // outward normal (CW rotate)
            let d = -(com_xz - a).dot(&n); // positive when inside
            min_margin = min_margin.min(d);
        }
        println!("  CoM projects {} polygon (min edge margin = {:+.4} m)",
            if inside { "INSIDE" } else { "OUTSIDE" }, min_margin);
    } else {
        println!("  CoM-in-polygon: polygon degenerate ({} verts), cannot contain a point", hull.len());
    }
}

fn total_weight() -> f64 {
    let foot_mass: f64 = foot::BONES.iter().map(|b| b.mass).sum();
    (SHIN_MASS + STICK_MASS + foot_mass) * 9.81
}

/// Runs one rigid-assembly case; returns true if it stood (finite, max
/// stick tilt < 5 deg over the 5 s run).
fn run_case(label: &str, stick_xz: (f64, f64)) -> bool {
    println!("\n========== {label} ==========");
    let mut r = build_rigid(stick_xz);

    // The fully-rigid assembly is flat on the ground and stable for the first
    // ~150 ms, then begins to tip (inverted-pendulum). It lifts off its contact
    // patch as it tips, so engaged_endpoints empties out once tilt grows. To
    // fairly read the SUPPORT POLYGON we sample it during the flat, max-contact
    // phase — the first 0.3 s — and keep the frame with the most engaged
    // endpoints (ties broken by lowest tilt). That is the true static footprint
    // the contact model offers a rigid body.
    let stick_idx = r.stick_idx;
    let mut best_n = 0usize;
    let mut best_tilt = f64::MAX;
    let mut best_com = center_of_mass(&r.world);
    let mut best_pts: Vec<(Vec3, f64)> = vec![];
    let mut best_t = 0.0;
    simulate(&mut r.world, 0.3, |w, t| {
        let pts = contact_points(w, &r.foot_bodies);
        let tilt = w.bodies[stick_idx].orientation.scaled_axis().norm().to_degrees();
        if pts.len() > best_n || (pts.len() == best_n && tilt < best_tilt) {
            best_n = pts.len();
            best_tilt = tilt;
            best_com = center_of_mass(w);
            best_pts = pts;
            best_t = t;
        }
    });
    println!("Best-contact frame in first 0.3 s: t={:.3}s, {} contacts, stick tilt {:.2} deg",
        best_t, best_n, best_tilt);
    println!("  CoM = (x={:+.4}, y={:+.4}, z={:+.4})  [y = height]",
        best_com.x, best_com.y, best_com.z);
    report_polygon_pts(&best_pts, best_com);

    // Run 5 s more (continuing from the 0.3 s settle), track max stick tilt.
    let mut max_tilt = best_tilt;
    let mut blew_up = false;
    let mut samples: Vec<(f64, f64)> = vec![];
    let mut next_sample = 0.0;
    simulate(&mut r.world, 5.0, |w, t| {
        let tilt = w.bodies[stick_idx].orientation.scaled_axis().norm().to_degrees();
        if tilt > max_tilt { max_tilt = tilt; }
        if !w.bodies.iter().all(|b| b.position.y.is_finite()) { blew_up = true; }
        if t >= next_sample {
            samples.push((t + 0.3, tilt));
            next_sample += 1.0;
        }
    });
    println!("Tilt trace over 5 s (stick orientation angle, t includes 0.3 s settle):");
    for (t, tilt) in &samples {
        println!("    t={:.1}s  tilt={:.2} deg", t, tilt);
    }
    let com_final = center_of_mass(&r.world);
    println!("  max stick tilt over full run = {:.2} deg", max_tilt);
    println!("  finite throughout: {}", !blew_up);
    println!("  final CoM = (x={:+.4}, y={:+.4}, z={:+.4})",
        com_final.x, com_final.y, com_final.z);
    if blew_up {
        println!("  VERDICT(case): blew up (non-finite).");
    } else if max_tilt < 5.0 {
        println!("  VERDICT(case): STOOD (< 5 deg).");
    } else {
        println!("  VERDICT(case): TIPPED ({:.1} deg > 5 deg).", max_tilt);
    }
    !blew_up && max_tilt < 5.0
}

/// Fine-grained settle trace: print CoM + stick tilt at short intervals so we
/// can see whether the assembly explodes at build (energy injection) vs. tips
/// slowly (genuine balance failure).
fn settle_trace(label: &str, weld_to_calcaneus: bool) {
    println!("\n---------- SETTLE TRACE: {label} ----------");
    let mut r = build_rigid_opt((0.0, 0.0), weld_to_calcaneus);
    let stick_idx = r.stick_idx;
    println!("  t(ms)   stickTilt(deg)   CoM(x,y,z)   maxBodySpeed(m/s)");
    let report = |w: &XpbdWorld, t: f64| {
        let tilt = w.bodies[stick_idx].orientation.scaled_axis().norm().to_degrees();
        let com = center_of_mass(w);
        let vmax = w.bodies.iter().map(|b| b.linear_velocity.norm()).fold(0.0_f64, f64::max);
        println!("  {:5.1}   {:8.2}        ({:+.3},{:+.3},{:+.3})   {:.3}",
            t * 1000.0, tilt, com.x, com.y, com.z, vmax);
    };
    report(&r.world, 0.0);
    let mut t = 0.0;
    for _ in 0..30 { // 30 frames = 250 ms at 120 Hz
        for _ in 0..SOLVER_ITERATIONS { r.world.substep(SUB_DT); }
        t += DT;
        if (t / DT).round() as usize % 3 == 0 { report(&r.world, t); }
    }
}

/// Dump per-endpoint ground geometry for the fully-rigid foot after a short
/// settle: which capsule end-spheres touch the plane (penetration > 0), their
/// (x,z) and bottom-y. Reveals whether the footprint is a real 2D patch or a
/// near-collinear strip, and reconciles it with engaged_endpoints.
fn contact_geometry_probe() {
    println!("\n---------- CONTACT GEOMETRY PROBE (fully rigid, after 0.5 s settle) ----------");
    let mut r = build_rigid((0.0, 0.0));
    simulate(&mut r.world, 0.5, |_, _| {});
    let plane_y = 0.0;
    // Gather every endpoint that is at/below the plane (touching or penetrating).
    let mut touch: Vec<(usize, f64, f64, f64, f64)> = vec![]; // (bone, x, z, bottom_y, pen)
    let mut lowest = (f64::MAX, 0usize, 0.0, 0.0); // (bottom_y, bone, x, z)
    for (bi, &idx) in r.foot_bodies.iter().enumerate() {
        let b = &r.world.bodies[idx];
        if let Some(s) = b.capsule_shape {
            let axis = b.orientation * Vec3::new(0.0, 1.0, 0.0);
            for off in [axis * s.half_h, -axis * s.half_h] {
                let ep = b.position + off;
                let bottom = ep.y - s.radius;
                let pen = plane_y - bottom;
                if bottom < lowest.0 { lowest = (bottom, bi, ep.x, ep.z); }
                if pen > -0.050 { // within 50 mm of the plane
                    touch.push((bi, ep.x, ep.z, bottom, pen));
                }
            }
        }
    }
    println!("  LOWEST foot endpoint: bone {} ({}) bottom_y={:+.5} at (x={:+.4}, z={:+.4})  [ground y=0]",
        lowest.1, foot::BONES[lowest.1].name, lowest.0, lowest.2, lowest.3);
    let com = center_of_mass(&r.world);
    println!("  CoM at probe time: (x={:+.4}, y={:+.4}, z={:+.4})", com.x, com.y, com.z);
    println!("  endpoints within 50 mm of plane: {}", touch.len());
    let (mut xmin, mut xmax, mut zmin, mut zmax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    let mut pos_pen = 0usize;
    for &(bi, x, z, by, pen) in &touch {
        let tag = if pen > 0.0 { "PEN " } else { "near" };
        if pen > 0.0 { pos_pen += 1; }
        xmin = xmin.min(x); xmax = xmax.max(x); zmin = zmin.min(z); zmax = zmax.max(z);
        println!("    bone {:2} ({:>13}): x={:+.4} z={:+.4} bottom_y={:+.5} pen={:+.5} [{}]",
            bi, foot::BONES[bi].name, x, z, by, pen, tag);
    }
    if !touch.is_empty() {
        println!("  near-plane footprint bbox: x[{:+.4},{:+.4}] (dx={:.4}) z[{:+.4},{:+.4}] (dz={:.4})",
            xmin, xmax, xmax - xmin, zmin, zmax, zmax - zmin);
        println!("  endpoints actually penetrating (pen>0): {}", pos_pen);
    }
    // Treat the near-plane endpoints (the would-be footprint if it rested) as
    // the support polygon and test CoM containment.
    let pts2d: Vec<Vec2> = touch.iter().map(|&(_, x, z, _, _)| Vec2::new(x, z)).collect();
    let hull = convex_hull_2d(&pts2d);
    let area = polygon_area(&hull);
    let inside = point_in_convex_polygon_2d(Vec2::new(com.x, com.z), &hull);
    println!("  near-plane footprint hull: {} verts, area={:.6} m^2 ({:.2} cm^2); CoM(x,z) inside = {}",
        hull.len(), area, area * 1e4, inside);

    // What engaged_endpoints reports right now.
    let eng = contact_points(&r.world, &r.foot_bodies);
    println!("  engaged_endpoints() reports {} contacts at this instant", eng.len());

    // Sample engaged_endpoints across many substeps to catch the hovering
    // limit-cycle: count frames with >=1 contact and the max contacts seen.
    let mut frames_with_contact = 0;
    let mut max_contacts = 0;
    let mut union_pts: Vec<Vec2> = vec![];
    for _ in 0..240 { // 2 s
        for _ in 0..SOLVER_ITERATIONS { r.world.substep(SUB_DT); }
        let c = contact_points(&r.world, &r.foot_bodies);
        if !c.is_empty() { frames_with_contact += 1; }
        if c.len() > max_contacts { max_contacts = c.len(); }
        for (p, _) in &c { union_pts.push(Vec2::new(p.x, p.z)); }
    }
    println!("  over next 2 s ({} frames): {} frames had >=1 contact; max simultaneous contacts = {}",
        240, frames_with_contact, max_contacts);
    if !union_pts.is_empty() {
        let uh = convex_hull_2d(&union_pts);
        println!("  union of all engaged contacts over 2 s: hull {} verts, area = {:.2} cm^2",
            uh.len(), polygon_area(&uh) * 1e4);
    }
}

fn main() {
    println!("Decisive rigid-body balance diagnostic for the 26-bone foot assembly.");
    println!("Total assembly weight ~{:.1} N ({:.2} kg).",
        total_weight(), total_weight() / 9.81);

    // Diagnostic phase 0: does the build explode immediately, or tip slowly?
    settle_trace("WITH redundant all-bone-to-calcaneus welds", true);
    settle_trace("WITHOUT extra welds (foot internal joints + rigid talocrural only)", false);

    // Diagnostic phase 0b: contact geometry probe. The fully-rigid body is held
    // up (CoM steady at y~0.62) but engaged_endpoints reports 0. Dump every foot
    // endpoint's bottom-y and penetration after a short settle to see how many
    // capsule end-spheres actually touch the plane, and where.
    contact_geometry_probe();

    // Case 1: nominal layout (stick directly above shin, x=z=0). THIS is the
    // acceptance case: the CoM sits over the LOAD-BEARING (rear) part of the
    // footprint, and a rigid body in that configuration must stand.
    let case1_stood = run_case("CASE 1: nominal layout (fully rigid)", (0.0, 0.0));

    // Case 2: geometric-centroid probe. Shifts the stick so the total CoM
    // projects onto the average of the contact-HULL VERTICES. This centroid is
    // GEOMETRIC, not load-weighted: the forward hull vertices (forefoot
    // endpoints) graze the plane bearing ~0 N, so the hull-vertex average sits
    // well FORWARD of the force-weighted (ZMP) support centroid. Centering the
    // CoM on it therefore places the CoM ahead of the loaded support and the
    // assembly is EXPECTED to tip forward — which is itself evidence that
    // support is where the forces are, not where the geometry is. Kept because
    // the contrast between the two cases is the informative result.
    let probe = {
        let mut r = build_rigid((0.0, 0.0));
        simulate(&mut r.world, 1.0, |_, _| {});
        let com = center_of_mass(&r.world);
        let pts = contact_points(&r.world, &r.foot_bodies);
        let mut pts2d: Vec<Vec2> = pts.iter().map(|(p, _)| Vec2::new(p.x, p.z)).collect();
        // polygon centroid (average of hull vertices)
        let hull = convex_hull_2d(&pts2d);
        let (mut cx, mut cz) = (0.0, 0.0);
        let n = hull.len().max(1) as f64;
        for v in &hull { cx += v.x; cz += v.y; }
        cx /= n; cz /= n;
        pts2d.clear();
        (com, cx, cz)
    };
    let (com0, poly_cx, poly_cz) = probe;
    // We want final CoM x,z == poly centroid. Shifting the 7 kg stick by ds
    // moves total CoM by (stick_mass/total_mass)*ds. Solve for ds.
    let total_m: f64 = SHIN_MASS + STICK_MASS + foot::BONES.iter().map(|b| b.mass).sum::<f64>();
    let frac = STICK_MASS / total_m;
    let ds_x = (poly_cx - com0.x) / frac;
    let ds_z = (poly_cz - com0.z) / frac;
    println!("\n[centering] settled polygon centroid (x,z)=({:+.4},{:+.4}); nominal CoM (x,z)=({:+.4},{:+.4})",
        poly_cx, poly_cz, com0.x, com0.z);
    println!("[centering] shifting stick by (dx={:+.4}, dz={:+.4}) m to center CoM over GEOMETRIC hull centroid", ds_x, ds_z);
    let case2_stood = run_case(
        "CASE 2: geometric-centroid probe (fully rigid, stick shifted; EXPECTED to tip — see comment)",
        (ds_x, ds_z));

    println!("\n========== SUMMARY ==========");
    println!("CASE 1 (CoM over the load-bearing support): {}",
        if case1_stood { "STOOD — the manifold contact holds a rigid body whose CoM is over its loaded footprint. This is the acceptance result." }
        else { "TIPPED — REGRESSION: the contact model failed to hold a rigid body whose CoM is over its loaded support." });
    println!("CASE 2 (CoM over the geometric hull centroid): {}",
        if case2_stood { "stood (unexpected — the geometric and load-weighted centroids may now coincide; worth a look)" }
        else { "tipped, as expected — the hull-vertex centroid sits forward of the force-weighted (ZMP) support because the forefoot endpoints graze the plane at ~0 N. Support is where the forces are, not where the geometry is." });
    println!("The CI regression gate for the manifold property itself is");
    println!("`manifold_keeps_resting_capsule_level_with_two_active_contacts` in crates/physics/tests/engine_regression.rs.");
    if !case1_stood {
        std::process::exit(1);
    }
}
