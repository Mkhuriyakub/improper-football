// Visual demo of Phase 3 step 2: bilateral standing via CoM-based
// whole-body balance. Same physics as `balance_bilateral.rs`, plus
// Bevy rendering and periodic disturbances so you can see the
// controller respond.
//
// Architecture (no hacks — composes proper engine primitives):
//   - `XpbdWorld::com_ground_projection` reads CoM
//   - `GroundContact::engaged_endpoints` reads contact points
//   - `convex_hull_2d` builds the support polygon
//   - `TorqueDispatcher` (with per-route limit) splits the KP·error
//     demand across both ankles
//   - 2× `AngularDamper` at the ankles provide implicit KD damping

use bevy::prelude::*;
use nalgebra::UnitQuaternion;

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::torque_dispatcher::{DispatchRoute, TorqueDispatcher};
use physics::math::{Vec2, Vec3 as NaVec3, convex_hull_2d, signed_distance_to_convex_polygon_2d};
use improper_football::render_util::rigid_body_to_transform;

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

const FOOT_MASS: f64 = 1.0;
const FOOT_HALF_H: f64 = 0.090;
const FOOT_RADIUS: f64 = 0.032;
const FOOT_ANCHOR_LOCAL: NaVec3 = NaVec3::new(0.0, 0.0, 0.0);

const SHIN_MASS: f64   = 3.0;
const SHIN_HALF_H: f64 = 0.21;
const SHIN_RADIUS: f64 = 0.05;

const THIGH_MASS: f64   = 7.0;
const THIGH_HALF_H: f64 = 0.225;
const THIGH_RADIUS: f64 = 0.07;

const PELVIS_MASS: f64   = 8.0;
const PELVIS_HALF_H: f64 = 0.10;
const PELVIS_RADIUS: f64 = 0.18;

const TORSO_MASS: f64   = 40.0;
const TORSO_HALF_H: f64 = 0.30;
const TORSO_RADIUS: f64 = 0.13;

const HIP_ANCHOR_PELVIS_X: f64 =  0.10;
const HIP_ANCHOR_PELVIS_Y: f64 = -0.05;

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;
const PF_K: f64 = 4.0;  const PF_C: f64 = 1.2;
const PF_MIN: f64 = -0.44;  const PF_MAX: f64 = 0.87;
const ANKLE_LIMIT_K: f64 = 80.0;
const ANKLE_LIMIT_EXP: f64 = 2.0;

const KNEE_JOINT_K: f64 = 80_000.0;
const KNEE_JOINT_C: f64 =    200.0;
const KNEE_PASSIVE_K: f64 = 5.0;  const KNEE_PASSIVE_C: f64 = 1.5;
const KNEE_MIN: f64 = -0.087;  const KNEE_MAX: f64 = 2.44;
const KNEE_LIMIT_K: f64 = 80.0;
const KNEE_LIMIT_EXP: f64 = 2.0;
const KNEE_PD_KP: f64 = 3000.0;  const KNEE_PD_KD: f64 = 300.0;

const HIP_JOINT_K: f64 = 200_000.0;
const HIP_JOINT_C: f64 =    400.0;
const HIP_FLEX_K: f64 = 10.0;  const HIP_FLEX_C: f64 = 7.0;
const HIP_ABD_K:  f64 = 10.0;  const HIP_ABD_C:  f64 = 7.0;
const HIP_ROT_K:  f64 =  3.0;  const HIP_ROT_C:  f64 = 0.40;
const HIP_FLEX_MIN: f64 = -0.35;  const HIP_FLEX_MAX: f64 = 2.09;
const HIP_ABD_MIN:  f64 = -0.52;  const HIP_ABD_MAX:  f64 = 0.78;
const HIP_ROT_MIN:  f64 = -0.61;  const HIP_ROT_MAX:  f64 = 0.87;
const HIP_LIMIT_K: f64 = 120.0;
const HIP_LIMIT_EXP: f64 = 2.0;
const HIP_FLEX_PD_KP: f64 = 4000.0;  const HIP_FLEX_PD_KD: f64 = 400.0;

const LUMBAR_JOINT_K: f64 = 300_000.0;
const LUMBAR_JOINT_C: f64 =    600.0;
const LUMBAR_PASSIVE_K: f64 = 150.0;  const LUMBAR_PASSIVE_C: f64 = 25.0;
const L_FLEX_MIN: f64 = -0.44;  const L_FLEX_MAX: f64 = 1.05;
const LUMBAR_LIMIT_K: f64 = 200.0;
const LUMBAR_LIMIT_EXP: f64 = 2.0;
const LUMBAR_PD_KP: f64 = 2500.0;  const LUMBAR_PD_KD: f64 = 250.0;

const BAL_KP_OFFSET: f64 = 800.0;
const PER_ANKLE_LIMIT: f64 = 14.0;
const ANKLE_KD: f64 = 120.0;

// No disturbances in this demo — the current controller architecture
// holds static balance cleanly but cannot recover from significant
// pushes (would need hip-strategy control or stepping, both deferred
// to later engine extensions). What you see: the body stands upright
// indefinitely. Live readout shows CoM, controller state, and
// support-polygon distance.

fn place_child(
    parent: &RigidBody, parent_distal: NaVec3, child_proximal: NaVec3,
    rel_rot: UnitQuaternion<f64>,
) -> (NaVec3, UnitQuaternion<f64>) {
    let child_orient = parent.orientation * rel_rot;
    let joint = parent.position + parent.orientation * parent_distal;
    let child_pos = joint - child_orient * child_proximal;
    (child_pos, child_orient)
}

fn add_elastic_joint(
    world: &mut XpbdWorld, parent: usize, child: usize,
    pa: NaVec3, ca: NaVec3, k: f64, c: f64,
) {
    let mut s = PositionalSpring::new(parent, child, k);
    s.local_anchor_a = pa; s.local_anchor_b = ca;
    world.positional_springs.push(s);
    let mut d = PositionalDamper::new(parent, child, c);
    d.local_anchor_a = pa; d.local_anchor_b = ca;
    world.positional_dampers.push(d);
}

#[derive(Clone, Copy, PartialEq)]
enum Side { Right, Left }
impl Side { fn sign(self) -> f64 { match self { Side::Right => 1.0, Side::Left => -1.0 } } }

#[derive(Clone, Copy)]
struct LegHandles { thigh_idx: usize, shin_idx: usize, foot_idx: usize }

fn add_leg(world: &mut XpbdWorld, pelvis_idx: usize, side: Side) -> LegHandles {
    let s = side.sign();
    let id = UnitQuaternion::identity();
    let foot_baseline = UnitQuaternion::from_scaled_axis(
        NaVec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let hip_anchor = NaVec3::new(HIP_ANCHOR_PELVIS_X * s, HIP_ANCHOR_PELVIS_Y, 0.0);

    let mut thigh = RigidBody::new_capsule(THIGH_MASS, THIGH_HALF_H, THIGH_RADIUS);
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        hip_anchor, NaVec3::new(0.0, THIGH_HALF_H, 0.0), id);
    thigh.position = pos; thigh.orientation = orient;
    let thigh_idx = world.add_body(thigh);

    add_elastic_joint(world, pelvis_idx, thigh_idx,
        hip_anchor, NaVec3::new(0.0, THIGH_HALF_H, 0.0),
        HIP_JOINT_K, HIP_JOINT_C);
    let hip_flex_axis = NaVec3::new(-1.0, 0.0, 0.0);
    let hip_abd_axis  = NaVec3::new(0.0, 0.0, s);
    let hip_rot_axis  = NaVec3::new(0.0, s, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(pelvis_idx, thigh_idx, hip_flex_axis,
        HIP_FLEX_MIN, HIP_FLEX_MAX, HIP_LIMIT_K, HIP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(pelvis_idx, thigh_idx, hip_flex_axis, HIP_FLEX_C));
    world.angular_springs.push(AngularSpring::new(pelvis_idx, thigh_idx, hip_flex_axis, HIP_FLEX_K, 0.0));
    world.angular_pd_controllers.push(AngularPDController::new(
        pelvis_idx, thigh_idx, hip_flex_axis, HIP_FLEX_PD_KP, HIP_FLEX_PD_KD));
    world.angular_soft_limits.push(AngularSoftLimit::new(pelvis_idx, thigh_idx, hip_abd_axis,
        HIP_ABD_MIN, HIP_ABD_MAX, HIP_LIMIT_K, HIP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(pelvis_idx, thigh_idx, hip_abd_axis, HIP_ABD_C));
    world.angular_springs.push(AngularSpring::new(pelvis_idx, thigh_idx, hip_abd_axis, HIP_ABD_K, 0.0));
    world.angular_soft_limits.push(AngularSoftLimit::new(pelvis_idx, thigh_idx, hip_rot_axis,
        HIP_ROT_MIN, HIP_ROT_MAX, HIP_LIMIT_K, HIP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(pelvis_idx, thigh_idx, hip_rot_axis, HIP_ROT_C));
    world.angular_springs.push(AngularSpring::new(pelvis_idx, thigh_idx, hip_rot_axis, HIP_ROT_K, 0.0));

    let mut shin = RigidBody::new_capsule(SHIN_MASS, SHIN_HALF_H, SHIN_RADIUS);
    let (pos, orient) = place_child(&world.bodies[thigh_idx],
        NaVec3::new(0.0, -THIGH_HALF_H, 0.0),
        NaVec3::new(0.0,  SHIN_HALF_H, 0.0), id);
    shin.position = pos; shin.orientation = orient;
    let shin_idx = world.add_body(shin);

    add_elastic_joint(world, thigh_idx, shin_idx,
        NaVec3::new(0.0, -THIGH_HALF_H, 0.0),
        NaVec3::new(0.0,  SHIN_HALF_H, 0.0),
        KNEE_JOINT_K, KNEE_JOINT_C);
    let knee_axis = NaVec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(thigh_idx, shin_idx, knee_axis,
        KNEE_MIN, KNEE_MAX, KNEE_LIMIT_K, KNEE_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(thigh_idx, shin_idx, knee_axis, KNEE_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(thigh_idx, shin_idx, knee_axis, KNEE_PASSIVE_K, 0.0));
    world.angular_pd_controllers.push(AngularPDController::new(
        thigh_idx, shin_idx, knee_axis, KNEE_PD_KP, KNEE_PD_KD));

    let mut foot = RigidBody::new_capsule(FOOT_MASS, FOOT_HALF_H, FOOT_RADIUS);
    let (pos, orient) = place_child(&world.bodies[shin_idx],
        NaVec3::new(0.0, -SHIN_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL, foot_baseline);
    foot.position = pos; foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(world, shin_idx, foot_idx,
        NaVec3::new(0.0, -SHIN_HALF_H, 0.0), FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);
    let pf_axis = NaVec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(shin_idx, foot_idx, pf_axis,
        PF_MIN, PF_MAX, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP).with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, pf_axis, PF_C));
    world.angular_springs.push(AngularSpring::new(shin_idx, foot_idx, pf_axis, PF_K, 0.0)
        .with_baseline(foot_baseline));

    // Active KD damping at the ankle for the balance controller (implicit).
    world.angular_dampers.push(AngularDamper::new(
        shin_idx, foot_idx, NaVec3::new(1.0, 0.0, 0.0), ANKLE_KD));

    LegHandles { thigh_idx, shin_idx, foot_idx }
}

/// Currently-engaged support polygon — a discrete signal (vertex
/// count jumps when an endpoint engages/disengages). Reported for
/// diagnostics; the controller uses a fixed target instead.
fn engaged_support_polygon(world: &XpbdWorld, foot_indices: &[usize]) -> Vec<Vec2> {
    let mut pts: Vec<Vec2> = Vec::new();
    for &foot_idx in foot_indices {
        for ep in world.engaged_endpoints(foot_idx).iter().flatten() {
            pts.push(Vec2::new(ep.contact_point.x, ep.contact_point.z));
        }
    }
    convex_hull_2d(&pts)
}

/// Geometric foot-axis polygon (continuous in foot pose). Used to
/// capture the controller's fixed reference target at startup.
fn geometric_foot_polygon(world: &XpbdWorld, foot_indices: &[usize], foot_half_h: f64) -> Vec<Vec2> {
    let mut pts: Vec<Vec2> = Vec::new();
    for &foot_idx in foot_indices {
        let (a, b) = world.bodies[foot_idx].axis_endpoints_world(foot_half_h);
        pts.push(Vec2::new(a.x, a.z));
        pts.push(Vec2::new(b.x, b.z));
    }
    convex_hull_2d(&pts)
}

fn polygon_centroid(poly: &[Vec2]) -> Vec2 {
    if poly.is_empty() { return Vec2::zeros(); }
    let sum: Vec2 = poly.iter().sum();
    sum / poly.len() as f64
}

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    sim_time: f64,
    leg_r: LegHandles,
    leg_l: LegHandles,
    pelvis_idx: usize,
    dispatcher_idx: usize,
    target_xz: Vec2,
    last_com_z: f64,
    last_demand: f64,
    last_supp_dist: f64,
}

#[derive(Component)]
struct BodyRef(usize);

#[derive(Component)]
struct Readout;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let id = UnitQuaternion::identity();

    let mut pelvis = RigidBody::new_capsule(PELVIS_MASS, PELVIS_HALF_H, PELVIS_RADIUS);
    pelvis.position = NaVec3::new(0.0, 0.952, 0.0);
    let pelvis_idx = world.add_body(pelvis);

    let mut torso = RigidBody::new_capsule(TORSO_MASS, TORSO_HALF_H, TORSO_RADIUS);
    let lumbar_anchor_pelvis = NaVec3::new(0.0, PELVIS_HALF_H, 0.0);
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        lumbar_anchor_pelvis, NaVec3::new(0.0, -TORSO_HALF_H, 0.0), id);
    torso.position = pos; torso.orientation = orient;
    let torso_idx = world.add_body(torso);

    add_elastic_joint(&mut world, pelvis_idx, torso_idx,
        lumbar_anchor_pelvis, NaVec3::new(0.0, -TORSO_HALF_H, 0.0),
        LUMBAR_JOINT_K, LUMBAR_JOINT_C);
    let lumbar_axis = NaVec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(pelvis_idx, torso_idx, lumbar_axis,
        L_FLEX_MIN, L_FLEX_MAX, LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(pelvis_idx, torso_idx, lumbar_axis, LUMBAR_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(pelvis_idx, torso_idx, lumbar_axis, LUMBAR_PASSIVE_K, 0.0));
    world.angular_pd_controllers.push(AngularPDController::new(
        pelvis_idx, torso_idx, lumbar_axis, LUMBAR_PD_KP, LUMBAR_PD_KD));

    let leg_r = add_leg(&mut world, pelvis_idx, Side::Right);
    let leg_l = add_leg(&mut world, pelvis_idx, Side::Left);

    // Ground contact via the world: every body with capsule_shape collides
    // with the ground at y=0. Pre-migration this binary disabled contact
    // for non-feet bodies via plane_y = -radius hacks.

    let dispatcher_idx = world.torque_dispatchers.len();
    world.torque_dispatchers.push(TorqueDispatcher::new(
        NaVec3::new(1.0, 0.0, 0.0),
        vec![
            DispatchRoute::new(leg_r.shin_idx, leg_r.foot_idx),
            DispatchRoute::new(leg_l.shin_idx, leg_l.foot_idx),
        ],
    ).with_per_route_limit(PER_ANKLE_LIMIT));

    // Materials & meshes.
    let foot_mat   = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.35, 0.25), ..default() });
    let leg_mat    = materials.add(StandardMaterial { base_color: Color::srgb(0.80, 0.62, 0.48), ..default() });
    let pelvis_mat = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.55, 0.62), ..default() });
    let torso_mat  = materials.add(StandardMaterial { base_color: Color::srgb(0.78, 0.62, 0.50), ..default() });

    let foot_mesh   = meshes.add(Capsule3d::new(FOOT_RADIUS as f32, (2.0 * FOOT_HALF_H) as f32));
    let shin_mesh   = meshes.add(Capsule3d::new(SHIN_RADIUS as f32, (2.0 * SHIN_HALF_H) as f32));
    let thigh_mesh  = meshes.add(Capsule3d::new(THIGH_RADIUS as f32, (2.0 * THIGH_HALF_H) as f32));
    let pelvis_mesh = meshes.add(Capsule3d::new(PELVIS_RADIUS as f32, (2.0 * PELVIS_HALF_H) as f32));
    let torso_mesh  = meshes.add(Capsule3d::new(TORSO_RADIUS as f32, (2.0 * TORSO_HALF_H) as f32));

    for leg in [leg_r, leg_l] {
        commands.spawn((PbrBundle { mesh: foot_mesh.clone(), material: foot_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[leg.foot_idx]), ..default() },
            BodyRef(leg.foot_idx)));
        commands.spawn((PbrBundle { mesh: shin_mesh.clone(), material: leg_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[leg.shin_idx]), ..default() },
            BodyRef(leg.shin_idx)));
        commands.spawn((PbrBundle { mesh: thigh_mesh.clone(), material: leg_mat.clone(),
            transform: rigid_body_to_transform(&world.bodies[leg.thigh_idx]), ..default() },
            BodyRef(leg.thigh_idx)));
    }
    commands.spawn((PbrBundle { mesh: pelvis_mesh, material: pelvis_mat,
        transform: rigid_body_to_transform(&world.bodies[pelvis_idx]), ..default() },
        BodyRef(pelvis_idx)));
    commands.spawn((PbrBundle { mesh: torso_mesh, material: torso_mat,
        transform: rigid_body_to_transform(&world.bodies[torso_idx]), ..default() },
        BodyRef(torso_idx)));

    let ground_mesh = meshes.add(Plane3d::default().mesh().size(4.0, 4.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0), ..default()
    });

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(2.8, 1.30, 2.8)
            .looking_at(Vec3::new(0.0, 0.7, 0.0), Vec3::Y),
        ..default()
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10_000.0, shadows_enabled: false, ..default() },
        transform: Transform::from_xyz(2.0, 4.0, 2.0)
            .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        ..default()
    });
    commands.insert_resource(AmbientLight {
        color: Color::WHITE, brightness: 220.0 });

    commands.spawn((TextBundle::from_section(
        "...",
        TextStyle { font_size: 18.0, color: Color::WHITE, ..default() },
    )
    .with_style(Style {
        position_type: PositionType::Absolute,
        top: Val::Px(12.0), left: Val::Px(12.0), ..default()
    }), Readout));

    let foot_indices = [leg_r.foot_idx, leg_l.foot_idx];
    let initial_polygon = geometric_foot_polygon(&world, &foot_indices, FOOT_HALF_H);
    let target_xz = polygon_centroid(&initial_polygon);

    commands.insert_resource(PhysicsWorld {
        world, sim_time: 0.0,
        leg_r, leg_l, pelvis_idx, dispatcher_idx,
        target_xz,
        last_com_z: 0.0, last_demand: 0.0, last_supp_dist: 0.0,
    });
    let _ = torso_idx;  // not needed without disturbance
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let leg_r = phys.leg_r;
    let leg_l = phys.leg_l;
    let dispatcher_idx = phys.dispatcher_idx;
    let target_xz = phys.target_xz;
    let foot_indices = [leg_r.foot_idx, leg_l.foot_idx];

    for _ in 0..SOLVER_ITERATIONS {
        // CoM-based controller with FIXED target (captured at startup).
        let com_xz = phys.world.com_ground_projection();
        let z_err = com_xz.y - target_xz.y;
        let demand = -BAL_KP_OFFSET * z_err;
        phys.world.torque_dispatchers[dispatcher_idx].demand = demand;

        phys.world.substep(SUB_DT);

        phys.last_com_z = com_xz.y;
        phys.last_demand = demand;
        // Diagnostic: distance to engaged support polygon.
        let engaged = engaged_support_polygon(&phys.world, &foot_indices);
        phys.last_supp_dist = signed_distance_to_convex_polygon_2d(com_xz, &engaged);
    }
    phys.sim_time += DT;
}

fn sync_transforms(
    phys: Res<PhysicsWorld>,
    mut q: Query<(&BodyRef, &mut Transform)>,
) {
    for (body_ref, mut t) in &mut q {
        *t = rigid_body_to_transform(&phys.world.bodies[body_ref.0]);
    }
}

fn update_readout(
    phys: Res<PhysicsWorld>,
    mut q: Query<&mut Text, With<Readout>>,
) {
    let pelvis_y = phys.world.bodies[phys.pelvis_idx].position.y;
    for mut text in &mut q {
        text.sections[0].value = format!(
            "t: {:.2} s\n\nPhase 3 step 2 — bilateral standing via CoM-based control.\n\npelvis y:   {:.3} m  (started at 0.952)\nCoM z:      {:+.4} m\ntarget z:   {:+.4} m  (fixed at startup)\nz error:    {:+.4} m\nsupp dist:  {:+.4} m  (positive = inside engaged polygon)\nankle τ:    {:+.2} N·m  (capped at ±{:.0} per ankle)\n\nThis demonstrates STATIC standing only. Disturbance recovery\nrequires hip-strategy control or stepping — both deferred\nto later engine extensions.\n\nNo hacks: TorqueDispatcher + 2× implicit AngularDampers.",
            phys.sim_time,
            pelvis_y,
            phys.last_com_z,
            phys.target_xz.y,
            phys.last_com_z - phys.target_xz.y,
            phys.last_supp_dist,
            phys.last_demand, PER_ANKLE_LIMIT,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Balance bilateral (Phase 3 step 2 — CoM-based whole-body control)".into(),
                resolution: (1100.0, 800.0).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Time::<Fixed>::from_hz(PHYSICS_HZ))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, physics_step)
        .add_systems(Update, (sync_transforms, update_readout))
        .run();
}
