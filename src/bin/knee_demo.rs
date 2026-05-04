// Visual demo of the knee on its own (no hip yet).
//
// Pinned vertical thigh + lower leg on a 1-DOF hinge. PD controller drives
// the knee through full anatomical ROM:
//
//   0–6 s     flexion sweep        Z: 0 → +2.20 → 0  (full curl, ~126°)
//   6–10 s    slight extension     Z: 0 → -0.05 → 0  (gentle hyperextension)
//   10–18 s   walking gait cycle   Z: 0 → +1.0 → 0   (running-style alternation)
//   18–26 s   slow continuous arc  large amplitude sweep mimicking a kick

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
use physics::math::{Vec3 as NaVec3, Mat3};
use improper_football::render_util::rigid_body_to_transform;

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

const THIGH_HALF_H: f64 = 0.225;
const THIGH_RADIUS: f64 = 0.070;

const LOWER_LEG_MASS:    f64 = 3.5;
const LOWER_LEG_RADIUS:  f64 = 0.050;
const LOWER_LEG_HALF_H:  f64 = 0.210;

const KNEE_JOINT_K: f64 = 80_000.0;
const KNEE_JOINT_C: f64 =    200.0;

// Passive biomech (less aggressive than validation since PD will dominate).
const KNEE_K:        f64 = 5.0;
const KNEE_C:        f64 = 1.5;
const KNEE_NEUTRAL:  f64 = 0.10;
const KNEE_MIN:      f64 = -0.087;
const KNEE_MAX:      f64 =  2.44;
const KNEE_LIMIT_K:  f64 =  80.0;
const KNEE_LIMIT_EXP:f64 =   2.0;

const KNEE_OFF_AXIS_K:        f64 = 12.0;
const KNEE_OFF_AXIS_C:        f64 = 0.80;
const KNEE_OFF_AXIS_LIM:      f64 = 0.30;
const KNEE_OFF_AXIS_LIMIT_K:  f64 = 50.0;

// Active drive — high-enough KP to overpower passive springs and track.
const KNEE_PD_KP: f64 = 300.0;
const KNEE_PD_KD: f64 =  20.0;

const THIGH_REST_POS: NaVec3 = NaVec3::new(0.0, 1.0, 0.0);

const SCHEDULE_PERIOD: f64 = 26.0;

fn capsule_inertia(mass: f64, radius: f64, half_h: f64) -> Mat3 {
    let h = 2.0 * half_h;
    let i_axial = 0.5 * mass * radius * radius;
    let i_trans = (1.0 / 12.0) * mass * (3.0 * radius * radius + h * h);
    Mat3::new(
        i_trans, 0.0,     0.0,
        0.0,     i_axial, 0.0,
        0.0,     0.0,     i_trans,
    )
}

fn place_child(
    parent: &RigidBody,
    parent_distal_local: NaVec3,
    child_proximal_local: NaVec3,
    rel_rot: UnitQuaternion<f64>,
) -> (NaVec3, UnitQuaternion<f64>) {
    let child_orient = parent.orientation * rel_rot;
    let joint_world = parent.position + parent.orientation * parent_distal_local;
    let child_pos = joint_world - child_orient * child_proximal_local;
    (child_pos, child_orient)
}

fn add_elastic_joint(
    world: &mut XpbdWorld,
    parent: usize, child: usize,
    parent_anchor: NaVec3, child_anchor: NaVec3,
    stiffness: f64, damping: f64,
) {
    let mut spring = PositionalSpring::new(parent, child, stiffness);
    spring.local_anchor_a = parent_anchor;
    spring.local_anchor_b = child_anchor;
    world.positional_springs.push(spring);

    let mut damp = PositionalDamper::new(parent, child, damping);
    damp.local_anchor_a = parent_anchor;
    damp.local_anchor_b = child_anchor;
    world.positional_dampers.push(damp);
}

fn add_axis(
    world: &mut XpbdWorld,
    parent: usize, child: usize,
    axis_local_a: NaVec3,
    min_angle: f64, max_angle: f64,
    stiffness: f64, damping: f64, neutral: f64,
    limit_stiffness: f64,
) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, axis_local_a,
        min_angle, max_angle, limit_stiffness, KNEE_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis_local_a, damping));
    world.angular_springs.push(AngularSpring::new(parent, child, axis_local_a, stiffness, neutral));
}

fn half_cosine(t: f64, duration: f64, peak: f64) -> f64 {
    let phase = (t / duration).clamp(0.0, 1.0);
    peak * 0.5 * (1.0 - (std::f64::consts::TAU * phase).cos())
}

fn knee_target(t: f64) -> (f64, &'static str) {
    let t = t.rem_euclid(SCHEDULE_PERIOD);
    if t < 6.0 {
        (half_cosine(t, 6.0, 2.20), "full flexion sweep")
    } else if t < 10.0 {
        // Gentle hyperextension probe — only slight excursion below 0.
        (half_cosine(t - 6.0, 4.0, -0.05), "hyperextension probe")
    } else if t < 18.0 {
        // Walking gait — knee flexes mid-swing then extends at heel strike.
        let local = t - 10.0;
        let phase = std::f64::consts::TAU * local / 1.0;  // 1 Hz cadence
        let envelope = (std::f64::consts::PI * local / 8.0).sin();
        let v = (0.50 + 0.50 * (-phase.cos() + 1.0)) * envelope;
        (v, "walking gait cycle")
    } else {
        // Slow large-amplitude kick.
        let local = t - 18.0;
        let envelope = (std::f64::consts::PI * local / 8.0).sin();
        let v = (1.50 * envelope.abs()) * envelope.signum();
        (v.max(0.0), "kick arc")
    }
}

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    thigh_idx: usize,
    lower_idx: usize,
    sim_time: f64,
    pd_idx: usize,
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

    let mut thigh = RigidBody::new(0.0, capsule_inertia(7.0, THIGH_RADIUS, THIGH_HALF_H));
    thigh.position = THIGH_REST_POS;
    thigh.orientation = UnitQuaternion::identity();
    let thigh_idx = world.add_body(thigh);

    let mut lower = RigidBody::new(LOWER_LEG_MASS,
        capsule_inertia(LOWER_LEG_MASS, LOWER_LEG_RADIUS, LOWER_LEG_HALF_H));
    let (pos, orient) = place_child(
        &world.bodies[thigh_idx],
        NaVec3::new(0.0, -THIGH_HALF_H, 0.0),
        NaVec3::new(0.0, LOWER_LEG_HALF_H, 0.0),
        UnitQuaternion::identity());
    lower.position = pos;
    lower.orientation = orient;
    let lower_idx = world.add_body(lower);

    add_elastic_joint(&mut world, thigh_idx, lower_idx,
        NaVec3::new(0.0, -THIGH_HALF_H, 0.0),
        NaVec3::new(0.0, LOWER_LEG_HALF_H, 0.0),
        KNEE_JOINT_K, KNEE_JOINT_C);

    let knee_axis = NaVec3::new(1.0, 0.0, 0.0);
    add_axis(&mut world, thigh_idx, lower_idx, knee_axis,
        KNEE_MIN, KNEE_MAX, KNEE_K, KNEE_C, KNEE_NEUTRAL, KNEE_LIMIT_K);
    add_axis(&mut world, thigh_idx, lower_idx, NaVec3::z(),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0, KNEE_OFF_AXIS_LIMIT_K);
    add_axis(&mut world, thigh_idx, lower_idx, NaVec3::y(),
        -KNEE_OFF_AXIS_LIM, KNEE_OFF_AXIS_LIM,
        KNEE_OFF_AXIS_K, KNEE_OFF_AXIS_C, 0.0, KNEE_OFF_AXIS_LIMIT_K);

    let pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thigh_idx, lower_idx, knee_axis, KNEE_PD_KP, KNEE_PD_KD)
            .with_target(KNEE_NEUTRAL));

    let thigh_mesh = meshes.add(Capsule3d::new(THIGH_RADIUS as f32, (2.0 * THIGH_HALF_H) as f32));
    let lower_mesh = meshes.add(Capsule3d::new(LOWER_LEG_RADIUS as f32, (2.0 * LOWER_LEG_HALF_H) as f32));

    let thigh_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.65, 0.50), ..default() });
    let lower_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.88, 0.68, 0.53), ..default() });

    commands.spawn((PbrBundle {
        mesh: thigh_mesh, material: thigh_mat,
        transform: rigid_body_to_transform(&world.bodies[thigh_idx]),
        ..default()
    }, BodyRef(thigh_idx)));
    commands.spawn((PbrBundle {
        mesh: lower_mesh, material: lower_mat,
        transform: rigid_body_to_transform(&world.bodies[lower_idx]),
        ..default()
    }, BodyRef(lower_idx)));

    let ground_mesh = meshes.add(Plane3d::default().mesh().size(3.0, 3.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });

    // Camera: side view so the sagittal-plane flex motion reads clearly.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(1.20, 0.85, 0.40)
            .looking_at(Vec3::new(0.0, 0.70, 0.0), Vec3::Y),
        ..default()
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10_000.0, shadows_enabled: true, ..default() },
        transform: Transform::from_xyz(2.0, 4.0, 2.0)
            .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        ..default()
    });
    commands.insert_resource(AmbientLight {
        color: Color::WHITE, brightness: 220.0 });

    commands.spawn((TextBundle::from_section(
        "...",
        TextStyle { font_size: 20.0, color: Color::WHITE, ..default() },
    )
    .with_style(Style {
        position_type: PositionType::Absolute,
        top: Val::Px(12.0), left: Val::Px(12.0),
        ..default()
    }), Readout));

    commands.insert_resource(PhysicsWorld {
        world, thigh_idx, lower_idx, sim_time: 0.0, pd_idx,
    });
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let (target, _) = knee_target(phys.sim_time);
    let pd_idx = phys.pd_idx;
    phys.world.angular_pd_controllers[pd_idx].target_angle = target;
    for _ in 0..SOLVER_ITERATIONS {
        phys.world.substep(SUB_DT);
    }
    phys.sim_time += DT;
}

fn sync_transforms(
    phys: Res<PhysicsWorld>,
    mut query: Query<(&BodyRef, &mut Transform)>,
) {
    for (body_ref, mut transform) in &mut query {
        let body = &phys.world.bodies[body_ref.0];
        *transform = rigid_body_to_transform(body);
    }
}

fn update_readout(
    phys: Res<PhysicsWorld>,
    mut query: Query<&mut Text, With<Readout>>,
) {
    let th = &phys.world.bodies[phys.thigh_idx];
    let lo = &phys.world.bodies[phys.lower_idx];
    let rel = (lo.orientation * th.orientation.inverse()).scaled_axis();
    let (target, phase) = knee_target(phys.sim_time);
    for mut text in &mut query {
        text.sections[0].value = format!(
            "{}\nt: {:.2} s  (cycle {:.0} s)\n\n  knee flex:  target {:+.2}  →  {:+.2} ({:+.0}°)",
            phase, phys.sim_time, SCHEDULE_PERIOD,
            target, rel.x, rel.x.to_degrees(),
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Knee demo (1-DOF hinge ROM)".into(),
                resolution: (1024.0, 768.0).into(),
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
