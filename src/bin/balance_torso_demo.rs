// Visual demo of Phase 3 step 1c: foot + shin + thigh + pelvis + torso
// chain (5 bodies, 4 articulations) balanced upright by an ankle
// absolute-tilt controller, with knee, hip, and lumbar held straight by
// posture PDs.
//
// Same physics as `balance_torso.rs`. Adds Bevy rendering and a periodic
// disturbance impulse so the controller's recovery is visible.

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
use physics::math::{Vec3 as NaVec3, swing_twist_angle};
use improper_football::render_util::rigid_body_to_transform;

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

const FOOT_MASS: f64 = 1.0;
const FOOT_HALF_H: f64 = 0.090;
const FOOT_RADIUS: f64 = 0.032;
const FOOT_ANCHOR_LOCAL: NaVec3 = NaVec3::new(0.0, FOOT_HALF_H, 0.0);

const SHIN_MASS: f64   = 3.0;
const SHIN_HALF_H: f64 = 0.21;
const SHIN_RADIUS: f64 = 0.05;

const THIGH_MASS: f64   = 7.0;
const THIGH_HALF_H: f64 = 0.225;
const THIGH_RADIUS: f64 = 0.07;

const PELVIS_MASS: f64   = 8.0;
const PELVIS_HALF_H: f64 = 0.10;
const PELVIS_RADIUS: f64 = 0.13;

const TORSO_MASS: f64   = 40.0;
const TORSO_HALF_H: f64 = 0.30;
const TORSO_RADIUS: f64 = 0.13;

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
const HIP_PASSIVE_K: f64 = 10.0;  const HIP_PASSIVE_C: f64 = 7.0;
const HIP_FLEX_MIN: f64 = -0.35;  const HIP_FLEX_MAX: f64 = 2.09;
const HIP_LIMIT_K: f64 = 120.0;
const HIP_LIMIT_EXP: f64 = 2.0;
const HIP_PD_KP: f64 = 4000.0;  const HIP_PD_KD: f64 = 400.0;

const LUMBAR_JOINT_K: f64 = 300_000.0;
const LUMBAR_JOINT_C: f64 =    600.0;
const LUMBAR_PASSIVE_K: f64 = 150.0;  const LUMBAR_PASSIVE_C: f64 = 25.0;
const L_FLEX_MIN: f64 = -0.44;  const L_FLEX_MAX: f64 = 1.05;
const LUMBAR_LIMIT_K: f64 = 200.0;
const LUMBAR_LIMIT_EXP: f64 = 2.0;
const LUMBAR_PD_KP: f64 = 2500.0;  const LUMBAR_PD_KD: f64 = 250.0;

const BAL_KP: f64 = 1000.0;
const BAL_KD: f64 =  235.0;

// Disturbance — every 1.8 s, add 2.5 rad/s of forward angular velocity
// to the torso. Sized so the body visibly tilts ~10° before the
// controller pulls it back. (The implicit-damping PD makes the chain
// extremely rigid; smaller shoves get absorbed too quickly to see.)
const DISTURBANCE_PERIOD: f64 = 1.8;
const DISTURBANCE_OMEGA:  f64 = 2.5;

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

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    sim_time: f64,
    next_disturbance: f64,
    foot_idx: usize,
    shin_idx: usize,
    thigh_idx: usize,
    pelvis_idx: usize,
    torso_idx: usize,
    last_thigh_tilt: f64,
    last_torso_tilt: f64,
    last_knee_bend: f64,
    last_hip_bend: f64,
    last_lumb_bend: f64,
    last_torque: f64,
    last_disturbance_time: f64,
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

    let mut torso = RigidBody::new_capsule(TORSO_MASS, TORSO_HALF_H, TORSO_RADIUS);
    torso.position = NaVec3::new(0.0, 1.402, 0.0);
    torso.orientation = id;
    let torso_idx = world.add_body(torso);

    let mut pelvis = RigidBody::new_capsule(PELVIS_MASS, PELVIS_HALF_H, PELVIS_RADIUS);
    let (pos, orient) = place_child(&world.bodies[torso_idx],
        NaVec3::new(0.0, -TORSO_HALF_H, 0.0),
        NaVec3::new(0.0,  PELVIS_HALF_H, 0.0), id);
    pelvis.position = pos; pelvis.orientation = orient;
    let pelvis_idx = world.add_body(pelvis);

    let mut thigh = RigidBody::new_capsule(THIGH_MASS, THIGH_HALF_H, THIGH_RADIUS);
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        NaVec3::new(0.0, -PELVIS_HALF_H, 0.0),
        NaVec3::new(0.0,  THIGH_HALF_H, 0.0), id);
    thigh.position = pos; thigh.orientation = orient;
    let thigh_idx = world.add_body(thigh);

    let mut shin = RigidBody::new_capsule(SHIN_MASS, SHIN_HALF_H, SHIN_RADIUS);
    let (pos, orient) = place_child(&world.bodies[thigh_idx],
        NaVec3::new(0.0, -THIGH_HALF_H, 0.0),
        NaVec3::new(0.0,  SHIN_HALF_H, 0.0), id);
    shin.position = pos; shin.orientation = orient;
    let shin_idx = world.add_body(shin);

    let foot_baseline = UnitQuaternion::from_scaled_axis(
        NaVec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    let mut foot = RigidBody::new_capsule(FOOT_MASS, FOOT_HALF_H, FOOT_RADIUS);
    let (pos, orient) = place_child(&world.bodies[shin_idx],
        NaVec3::new(0.0, -SHIN_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL, foot_baseline);
    foot.position = pos; foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    // Lumbar.
    add_elastic_joint(&mut world, torso_idx, pelvis_idx,
        NaVec3::new(0.0, -TORSO_HALF_H, 0.0),
        NaVec3::new(0.0,  PELVIS_HALF_H, 0.0),
        LUMBAR_JOINT_K, LUMBAR_JOINT_C);
    let lumbar_axis = NaVec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(
        AngularSoftLimit::new(torso_idx, pelvis_idx, lumbar_axis,
            L_FLEX_MIN, L_FLEX_MAX, LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP));
    world.angular_dampers.push(
        AngularDamper::new(torso_idx, pelvis_idx, lumbar_axis, LUMBAR_PASSIVE_C));
    world.angular_springs.push(
        AngularSpring::new(torso_idx, pelvis_idx, lumbar_axis, LUMBAR_PASSIVE_K, 0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(torso_idx, pelvis_idx, lumbar_axis,
            LUMBAR_PD_KP, LUMBAR_PD_KD));

    // Hip.
    add_elastic_joint(&mut world, pelvis_idx, thigh_idx,
        NaVec3::new(0.0, -PELVIS_HALF_H, 0.0),
        NaVec3::new(0.0,  THIGH_HALF_H, 0.0),
        HIP_JOINT_K, HIP_JOINT_C);
    let hip_axis = NaVec3::new(-1.0, 0.0, 0.0);
    world.angular_soft_limits.push(
        AngularSoftLimit::new(pelvis_idx, thigh_idx, hip_axis,
            HIP_FLEX_MIN, HIP_FLEX_MAX, HIP_LIMIT_K, HIP_LIMIT_EXP));
    world.angular_dampers.push(
        AngularDamper::new(pelvis_idx, thigh_idx, hip_axis, HIP_PASSIVE_C));
    world.angular_springs.push(
        AngularSpring::new(pelvis_idx, thigh_idx, hip_axis, HIP_PASSIVE_K, 0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thigh_idx, hip_axis,
            HIP_PD_KP, HIP_PD_KD));

    // Knee.
    add_elastic_joint(&mut world, thigh_idx, shin_idx,
        NaVec3::new(0.0, -THIGH_HALF_H, 0.0),
        NaVec3::new(0.0,  SHIN_HALF_H, 0.0),
        KNEE_JOINT_K, KNEE_JOINT_C);
    let knee_axis = NaVec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(
        AngularSoftLimit::new(thigh_idx, shin_idx, knee_axis,
            KNEE_MIN, KNEE_MAX, KNEE_LIMIT_K, KNEE_LIMIT_EXP));
    world.angular_dampers.push(
        AngularDamper::new(thigh_idx, shin_idx, knee_axis, KNEE_PASSIVE_C));
    world.angular_springs.push(
        AngularSpring::new(thigh_idx, shin_idx, knee_axis, KNEE_PASSIVE_K, 0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(thigh_idx, shin_idx, knee_axis,
            KNEE_PD_KP, KNEE_PD_KD));

    // Ankle.
    add_elastic_joint(&mut world, shin_idx, foot_idx,
        NaVec3::new(0.0, -SHIN_HALF_H, 0.0), FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);
    let pf_axis = NaVec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(
        AngularSoftLimit::new(shin_idx, foot_idx, pf_axis,
            PF_MIN, PF_MAX, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP)
            .with_baseline(foot_baseline));
    world.angular_dampers.push(
        AngularDamper::new(shin_idx, foot_idx, pf_axis, PF_C));
    world.angular_springs.push(
        AngularSpring::new(shin_idx, foot_idx, pf_axis, PF_K, 0.0)
            .with_baseline(foot_baseline));

    // Ground contact via the world: every body with capsule_shape collides
    // with the ground at y=0. Pre-migration shin/thigh/pelvis/torso had
    // per-body disable hacks; now they collide if the chain falls.

    // Meshes & materials.
    let foot_mat   = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.35, 0.25), ..default() });
    let leg_mat    = materials.add(StandardMaterial { base_color: Color::srgb(0.80, 0.62, 0.48), ..default() });
    let pelvis_mat = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.55, 0.62), ..default() });
    let torso_mat  = materials.add(StandardMaterial { base_color: Color::srgb(0.78, 0.62, 0.50), ..default() });

    let foot_mesh   = meshes.add(Capsule3d::new(FOOT_RADIUS as f32, (2.0 * FOOT_HALF_H) as f32));
    let shin_mesh   = meshes.add(Capsule3d::new(SHIN_RADIUS as f32, (2.0 * SHIN_HALF_H) as f32));
    let thigh_mesh  = meshes.add(Capsule3d::new(THIGH_RADIUS as f32, (2.0 * THIGH_HALF_H) as f32));
    let pelvis_mesh = meshes.add(Capsule3d::new(PELVIS_RADIUS as f32, (2.0 * PELVIS_HALF_H) as f32));
    let torso_mesh  = meshes.add(Capsule3d::new(TORSO_RADIUS as f32, (2.0 * TORSO_HALF_H) as f32));

    commands.spawn((PbrBundle { mesh: foot_mesh, material: foot_mat,
        transform: rigid_body_to_transform(&world.bodies[foot_idx]), ..default() },
        BodyRef(foot_idx)));
    commands.spawn((PbrBundle { mesh: shin_mesh, material: leg_mat.clone(),
        transform: rigid_body_to_transform(&world.bodies[shin_idx]), ..default() },
        BodyRef(shin_idx)));
    commands.spawn((PbrBundle { mesh: thigh_mesh, material: leg_mat,
        transform: rigid_body_to_transform(&world.bodies[thigh_idx]), ..default() },
        BodyRef(thigh_idx)));
    commands.spawn((PbrBundle { mesh: pelvis_mesh, material: pelvis_mat,
        transform: rigid_body_to_transform(&world.bodies[pelvis_idx]), ..default() },
        BodyRef(pelvis_idx)));
    commands.spawn((PbrBundle { mesh: torso_mesh, material: torso_mat,
        transform: rigid_body_to_transform(&world.bodies[torso_idx]), ..default() },
        BodyRef(torso_idx)));

    // Ground.
    let ground_mesh = meshes.add(Plane3d::default().mesh().size(4.0, 4.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0), ..default()
    });

    // Camera framed for the chain (top of torso ~ y=1.7).
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(2.8, 1.40, 2.8)
            .looking_at(Vec3::new(0.0, 0.85, 0.0), Vec3::Y),
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

    commands.insert_resource(PhysicsWorld {
        world, sim_time: 0.0,
        next_disturbance: DISTURBANCE_PERIOD,
        foot_idx, shin_idx, thigh_idx, pelvis_idx, torso_idx,
        last_thigh_tilt: 0.0, last_torso_tilt: 0.0,
        last_knee_bend: 0.0, last_hip_bend: 0.0, last_lumb_bend: 0.0,
        last_torque: 0.0,
        last_disturbance_time: -1.0,
    });
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    if phys.sim_time >= phys.next_disturbance {
        let torso_idx = phys.torso_idx;
        phys.world.bodies[torso_idx].angular_velocity.x += DISTURBANCE_OMEGA;
        phys.last_disturbance_time = phys.sim_time;
        phys.next_disturbance += DISTURBANCE_PERIOD;
    }

    let shin_idx = phys.shin_idx;
    let foot_idx = phys.foot_idx;
    let thigh_idx = phys.thigh_idx;
    let torso_idx = phys.torso_idx;
    let pelvis_idx = phys.pelvis_idx;

    for _ in 0..SOLVER_ITERATIONS {
        // Sense thigh (closest-to-actuator rigid segment, see project memory).
        let thigh = &phys.world.bodies[thigh_idx];
        let thigh_tilt_x  = thigh.orientation.scaled_axis().x;
        let thigh_omega_x = thigh.angular_velocity.x;
        let tau_x = -BAL_KP * thigh_tilt_x - BAL_KD * thigh_omega_x;
        let tau_world = NaVec3::new(tau_x, 0.0, 0.0);
        phys.world.bodies[shin_idx].apply_torque(tau_world);
        phys.world.bodies[foot_idx].apply_torque(-tau_world);

        phys.world.substep(SUB_DT);

        phys.last_thigh_tilt = thigh_tilt_x;
        phys.last_torque = tau_x;
    }
    phys.sim_time += DT;

    phys.last_torso_tilt = phys.world.bodies[torso_idx].orientation.scaled_axis().x;

    let q_shin_in_thigh = phys.world.bodies[thigh_idx].orientation.inverse()
                        * phys.world.bodies[shin_idx].orientation;
    phys.last_knee_bend = swing_twist_angle(q_shin_in_thigh, NaVec3::new(1.0, 0.0, 0.0));

    let q_thigh_in_pelvis = phys.world.bodies[pelvis_idx].orientation.inverse()
                          * phys.world.bodies[thigh_idx].orientation;
    phys.last_hip_bend = swing_twist_angle(q_thigh_in_pelvis, NaVec3::new(-1.0, 0.0, 0.0));

    let q_pelvis_in_torso = phys.world.bodies[torso_idx].orientation.inverse()
                          * phys.world.bodies[pelvis_idx].orientation;
    phys.last_lumb_bend = swing_twist_angle(q_pelvis_in_torso, NaVec3::new(1.0, 0.0, 0.0));
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
    let since_disturb = if phys.last_disturbance_time >= 0.0 {
        phys.sim_time - phys.last_disturbance_time
    } else {
        f64::INFINITY
    };
    let next_in = (phys.next_disturbance - phys.sim_time).max(0.0);
    for mut text in &mut q {
        text.sections[0].value = format!(
            "t: {:.2} s\n\nPhase 3 step 1c — 4-articulation chain balanced.\n\nthigh tilt:  {:+.3}°  (sensed by controller)\ntorso tilt:  {:+.3}°  (top of chain)\nknee bend:   {:+.3}°\nhip bend:    {:+.3}°\nlumbar bend: {:+.3}°\nankle tau:   {:+.1} N·m\n\nDisturbance every {:.1}s (forward shove on torso).\nLast disturbance: {:.1}s ago. Next in {:.1}s.\n\nKnee/hip/lumbar held by posture PDs ({:.0}/{:.0}/{:.0}).\nAnkle drives back to vertical from THIGH tilt.",
            phys.sim_time,
            phys.last_thigh_tilt.to_degrees(),
            phys.last_torso_tilt.to_degrees(),
            phys.last_knee_bend.to_degrees(),
            phys.last_hip_bend.to_degrees(),
            phys.last_lumb_bend.to_degrees(),
            phys.last_torque,
            DISTURBANCE_PERIOD,
            since_disturb.min(999.0),
            next_in,
            KNEE_PD_KP, HIP_PD_KP, LUMBAR_PD_KP,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Balance torso (Phase 3 step 1c — 4-articulation chain)".into(),
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
