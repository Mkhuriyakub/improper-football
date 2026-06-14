// Anatomical ankle — text validation binary (26-bone foot).
//
// Builds the shin + 26-bone anatomical foot (via the foot module) plus a
// load stick, and runs PASS/FAIL static-balance tests. The foot is the
// `improper_football::anatomy::foot` assembly; this host only adds the
// shin, the talocrural joint (shin↔talus), and the load stick.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::math::Vec3;
use nalgebra::UnitQuaternion;

use improper_football::anatomy::foot::{self, FootHandles};

// ─── Simulation timing ────────────────────────────────────────────────

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

// ─── Body parameters ──────────────────────────────────────────────────

const SHIN_MASS:    f64 = 3.00;
const SHIN_HALF_H:  f64 = 0.21;
const SHIN_RADIUS:  f64 = 0.05;

const STICK_MASS:    f64 = 7.00;  // load above shin; total above-ankle ≈ 10 kg
const STICK_HALF_H:  f64 = 0.40;
const STICK_RADIUS:  f64 = 0.08;

// ─── Assembly ─────────────────────────────────────────────────────────

struct Assembly {
    world: XpbdWorld,
    #[allow(dead_code)]
    foot: FootHandles,
    #[allow(dead_code)]
    shin_idx: usize,
    stick_idx: usize,
    #[allow(dead_code)]
    talocrural_pd_idx: usize,
}

fn build_world() -> Assembly {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let id = UnitQuaternion::identity();

    // Talus attaches here; tuned so the flat sole rests on the ground.
    // Foot origin = attach.y − talus_rest.y (0.057); the sole plane is at
    // 0.018 above origin, so attach.y = 0.057 − 0.018 = 0.039 puts the sole
    // exactly on y=0. A hair higher lets it settle into contact.
    let attach = Vec3::new(0.0, 0.039, 0.0);

    // Shin: vertical, distal (bottom) end at the talus attach point.
    let mut shin = RigidBody::new_capsule(SHIN_MASS, SHIN_HALF_H, SHIN_RADIUS);
    shin.position = Vec3::new(0.0, attach.y + SHIN_HALF_H, 0.0);
    shin.orientation = id;
    let shin_idx = world.add_body(shin);

    // Foot (26 bones); talus center at `attach`.
    let foot = foot::build_foot(&mut world, attach);

    // Diagnostic: optionally pin every foot bone immovable. This isolates
    // the ankle joint from the foot-on-ground rocking mode. With the foot
    // pinned the ankle PD balances the 11 kg column indefinitely (PASS),
    // which is the key evidence that the standing-balance failure is the
    // foot rocking off its small contact patch, NOT an ankle-gain problem.
    if std::env::var("ANKLE_PIN_FOOT").is_ok() {
        for &idx in foot.bodies.iter() {
            world.bodies[idx].inv_mass = 0.0;
            world.bodies[idx].inv_inertia_tensor = nalgebra::Matrix3::zeros();
        }
    }

    // Talocrural joint shin↔talus (baseline-aware; DF 20°/PF 40°).
    let talocrural_pd_idx = foot::attach_talocrural(
        &mut world, shin_idx, &foot, -0.35, 0.70, 800.0, 40.0);

    // Load stick welded to the shin top (both identity-oriented, so the
    // foot's weld with baseline-free AngularLimitConstraints is valid).
    let mut stick = RigidBody::new_capsule(STICK_MASS, STICK_HALF_H, STICK_RADIUS);
    stick.position = Vec3::new(0.0, attach.y + 2.0 * SHIN_HALF_H + STICK_HALF_H, 0.0);
    stick.orientation = id;
    let stick_idx = world.add_body(stick);
    let weld_pt = Vec3::new(0.0, attach.y + 2.0 * SHIN_HALF_H, 0.0); // shin top
    let anchor_shin = foot::anchor_at(&world, shin_idx, weld_pt);
    let anchor_stick = foot::anchor_at(&world, stick_idx, weld_pt);
    foot::weld(&mut world, shin_idx, stick_idx, anchor_shin, anchor_stick);

    Assembly { world, foot, shin_idx, stick_idx, talocrural_pd_idx }
}

// ─── Test framework ───────────────────────────────────────────────────

struct TestResult {
    name: &'static str,
    passed: bool,
    message: String,
}

impl TestResult {
    fn pass(name: &'static str) -> Self {
        Self { name, passed: true, message: String::from("PASS") }
    }
    fn fail(name: &'static str, message: String) -> Self {
        Self { name, passed: false, message: format!("FAIL: {message}") }
    }
}

/// Run the simulation for `duration` seconds, calling `per_frame` once per
/// frame (after each `SOLVER_ITERATIONS` substeps).
fn simulate(
    world: &mut XpbdWorld,
    duration: f64,
    mut per_frame: impl FnMut(&mut XpbdWorld, f64),
) {
    let mut t = 0.0;
    while t < duration {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;
        per_frame(world, t);
    }
}

fn test_sustained_stability() -> TestResult {
    let mut a = build_world();
    let mut blew_up = false;
    simulate(&mut a.world, 10.0, |w, _t| {
        if !w.bodies.iter().all(|b| b.position.y.is_finite()) { blew_up = true; }
    });
    if blew_up { return TestResult::fail("sustained stability", "non-finite position".into()); }
    TestResult::pass("sustained stability")
}

fn test_static_balance() -> TestResult {
    let mut a = build_world();
    let mut max_tilt = 0.0_f64;
    simulate(&mut a.world, 5.0, |w, _t| {
        let tilt = w.bodies[a.stick_idx].orientation.scaled_axis().norm();
        if tilt > max_tilt { max_tilt = tilt; }
    });
    let deg = max_tilt.to_degrees();
    if deg < 5.0 { TestResult::pass("static balance") }
    else { TestResult::fail("static balance", format!("max tilt {deg:.2}° (limit 5°)")) }
}

fn diagnose() {
    let mut a = build_world();
    let mut tprev = 0.0;
    for &tstop in &[0.30_f64, 0.50, 1.0, 2.0] {
        simulate(&mut a.world, tstop - tprev, |_, _| {});
        tprev = tstop;
        let sa = a.world.bodies[a.stick_idx].orientation.scaled_axis();
        let (mut m, mut c) = (0.0, Vec3::zeros());
        for b in &a.world.bodies { let mm = 1.0/b.inv_mass; m += mm; c += b.position * mm; }
        c /= m;
        let mut pts: Vec<(f64, f64, f64)> = vec![];
        for idx in 0..a.world.bodies.len() {
            for e in a.world.engaged_endpoints(idx).iter().flatten() {
                pts.push((e.contact_point.x, e.contact_point.z, e.normal_force));
            }
        }
        let (mut xmin, mut xmax, mut zmin, mut zmax) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        let mut ftot = 0.0;
        for &(x, z, f) in &pts {
            xmin = xmin.min(x); xmax = xmax.max(x);
            zmin = zmin.min(z); zmax = zmax.max(z); ftot += f;
        }
        println!("--- t≈{tstop:.2}s tilt={:.1}° (x={:.3} y={:.3} z={:.3}) CoM=({:.4},{:.4},{:.4})",
            sa.norm().to_degrees(), sa.x, sa.y, sa.z, c.x, c.y, c.z);
        println!("    {} contacts, Fn_total={ftot:.0}N, polygon x[{xmin:.3},{xmax:.3}] z[{zmin:.3},{zmax:.3}]; CoM inside-x={} inside-z={}",
            pts.len(), c.x >= xmin && c.x <= xmax, c.z >= zmin && c.z <= zmax);
    }
}

fn main() {
    if std::env::args().any(|a| a == "--diagnose") { diagnose(); return; }
    let mut results = vec![];
    results.push(test_sustained_stability());
    results.push(test_static_balance());
    for r in &results {
        println!("{}: {}", r.name, r.message);
    }
    if results.iter().any(|r| !r.passed) {
        eprintln!(
            "\nNote: passive static balance is a known structural limit — an \
             11 kg load with CoM ~0.6 m up cannot stand on a realistic ~5x15 cm \
             foot without active CoP/whole-body control. Run with ANKLE_PIN_FOOT=1 \
             to confirm the ankle PD itself balances the column (PASS); the failure \
             is the foot rocking off its small contact patch, not an ankle-gain issue. \
             Run with --diagnose for the tilt/CoM/support-polygon trace."
        );
        std::process::exit(1);
    }
}
