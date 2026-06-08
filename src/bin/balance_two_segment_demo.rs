// Visual demo of Phase 3 step 1b: foot + shin + thigh-lump balanced
// upright by an ankle absolute-tilt controller, with the knee held
// straight by a posture PD.
//
// Same physics as `balance_two_segment.rs`. Adds Bevy rendering and a
// periodic disturbance impulse so the controller's recovery behavior is
// visible — without disturbances the chain just stands still and the
// demo looks like a static screenshot.
//
// Disturbance: every 2.0 s, an impulse equivalent to ~5 N·m·s of
// forward angular velocity is added to the thigh's angular velocity.
// The controller catches the chain and brings it back to vertical.

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

const THIGH_MASS: f64   = 47.0;
const THIGH_HALF_H: f64 = 0.29;
const THIGH_RADIUS: f64 = 0.07;

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;

const PF_K: f64 = 4.0;  const PF_C: f64 = 1.2;
const PF_MIN: f64 = -0.44;
const PF_MAX: f64 =  0.87;
const ANKLE_LIMIT_K: f64 = 80.0;
const ANKLE_LIMIT_EXP: f64 = 2.0;

const KNEE_JOINT_K: f64 = 80_000.0;
const KNEE_JOINT_C: f64 =    200.0;
const KNEE_PASSIVE_K: f64 = 5.0;  const KNEE_PASSIVE_C: f64 = 1.5;
const KNEE_MIN: f64 = -0.087;  const KNEE_MAX: f64 = 2.44;
const KNEE_LIMIT_K: f64 = 80.0;
const KNEE_LIMIT_EXP: f64 = 2.0;

const KNEE_PD_KP: f64 = 2000.0;
const KNEE_PD_KD: f64 =  200.0;

const BAL_KP: f64 = 600.0;
const BAL_KD: f64 =  60.0;

// Disturbance: every DISTURBANCE_PERIOD seconds, add this amount of
// instantaneous angular velocity to the thigh about world +x.
// 1.0 rad/s on a thigh+lump ≈ a sharp shove.
const DISTURBANCE_PERIOD: f64 = 2.0;
const DISTURBANCE_OMEGA:  f64 = 1.0;

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
    last_thigh_tilt: f64,
    last_knee_bend: f64,
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

    let mut thigh = RigidBody::new_capsule(THIGH_MASS, THIGH_HALF_H, THIGH_RADIUS);
    thigh.position = NaVec3::new(0.0, 0.452 + THIGH_HALF_H, 0.0);
    thigh.orientation = id;
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
    // with the ground at y=0. Shin/thigh now collide if the chain tips far
    // enough; pre-migration they had per-body disable hacks.

    // Meshes & materials.
    let foot_mat  = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.35, 0.25), ..default() });
    let shin_mat  = materials.add(StandardMaterial { base_color: Color::srgb(0.80, 0.62, 0.48), ..default() });
    let thigh_mat = materials.add(StandardMaterial { base_color: Color::srgb(0.78, 0.62, 0.50), ..default() });

    let foot_mesh  = meshes.add(Capsule3d::new(FOOT_RADIUS as f32, (2.0 * FOOT_HALF_H) as f32));
    let shin_mesh  = meshes.add(Capsule3d::new(SHIN_RADIUS as f32, (2.0 * SHIN_HALF_H) as f32));
    let thigh_mesh = meshes.add(Capsule3d::new(THIGH_RADIUS as f32, (2.0 * THIGH_HALF_H) as f32));

    commands.spawn((PbrBundle { mesh: foot_mesh, material: foot_mat,
        transform: rigid_body_to_transform(&world.bodies[foot_idx]), ..default() },
        BodyRef(foot_idx)));
    commands.spawn((PbrBundle { mesh: shin_mesh, material: shin_mat,
        transform: rigid_body_to_transform(&world.bodies[shin_idx]), ..default() },
        BodyRef(shin_idx)));
    commands.spawn((PbrBundle { mesh: thigh_mesh, material: thigh_mat,
        transform: rigid_body_to_transform(&world.bodies[thigh_idx]), ..default() },
        BodyRef(thigh_idx)));

    // Ground.
    let ground_mesh = meshes.add(Plane3d::default().mesh().size(4.0, 4.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0), ..default()
    });

    // Camera framed for the chain (top of thigh-lump ≈ y=1.03 m).
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(2.5, 1.10, 2.5)
            .looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
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
        foot_idx, shin_idx, thigh_idx,
        last_thigh_tilt: 0.0, last_knee_bend: 0.0, last_torque: 0.0,
        last_disturbance_time: -1.0,
    });
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    // Disturbance: at scheduled times, add angular velocity to the thigh
    // about world +x (forward shove). This is applied directly to the
    // velocity, not as a torque, so it's a true impulse — controller
    // gets to demonstrate recovery.
    if phys.sim_time >= phys.next_disturbance {
        let thigh_idx = phys.thigh_idx;
        phys.world.bodies[thigh_idx].angular_velocity.x += DISTURBANCE_OMEGA;
        phys.last_disturbance_time = phys.sim_time;
        phys.next_disturbance += DISTURBANCE_PERIOD;
    }

    let shin_idx = phys.shin_idx;
    let foot_idx = phys.foot_idx;
    let thigh_idx = phys.thigh_idx;

    // Controller fires inside the substep loop (XpbdWorld clears
    // accumulators each substep — see project memory).
    for _ in 0..SOLVER_ITERATIONS {
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

    let q_shin_in_thigh = phys.world.bodies[thigh_idx].orientation.inverse()
                        * phys.world.bodies[shin_idx].orientation;
    phys.last_knee_bend = swing_twist_angle(q_shin_in_thigh, NaVec3::new(1.0, 0.0, 0.0));
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
            "t: {:.2} s\n\nPhase 3 step 1b — articulated 2-segment pendulum balanced.\n\nthigh tilt:  {:+.3}° (max ~5.7° goal)\nknee bend:   {:+.3}° (locked by posture PD)\nankle tau:   {:+.1} N·m\n\nDisturbance every {:.1}s (forward shove on thigh).\nLast disturbance: {:.1}s ago. Next in {:.1}s.\n\nKnee held straight by KP={:.0}, KD={:.0} posture PD.\nAnkle drives back to vertical via absolute-tilt torque.",
            phys.sim_time,
            phys.last_thigh_tilt.to_degrees(),
            phys.last_knee_bend.to_degrees(),
            phys.last_torque,
            DISTURBANCE_PERIOD,
            since_disturb.min(999.0),
            next_in,
            KNEE_PD_KP, KNEE_PD_KD,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Balance two-segment (Phase 3 step 1b — ankle balance with locked knee)".into(),
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
