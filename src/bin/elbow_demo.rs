// Visual demo of the biomechanical elbow.
//
// Renders the two-body elbow rig (upper arm + lower arm) with all the joint
// primitives we've validated — passive damping, passive stiffness, soft
// limit, and an active PD controller — running in real time.
//
// The active controller's target_angle cycles through the same step schedule
// as src/bin/elbow_active_step.rs:
//   t = 0.0 → 1.0:  target = 0.30 rad
//   t = 1.0 → 2.5:  target = 0.50 rad
//   t = 2.5 → 4.0:  target = 1.00 rad
//   t = 4.0 → 5.5:  target = 0.30 rad
//   t = 5.5 → 7.0:  target = 0.00 rad
// then loops.
//
// What you should see: the lower arm cycling through commanded targets,
// settling to the analytical equilibrium for each (offset slightly from the
// target by gravity + passive resistance), with realistic motion speed and
// no overshoot.

use bevy::prelude::*;
use nalgebra::UnitQuaternion;

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_constraint::AngularLimitConstraint;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::angular_pd_controller::AngularPDController;
use physics::math::Vec3 as NaVec3;
use player::segment::{BodyType, SegmentId, segment_def};
use improper_football::render_util::rigid_body_to_transform;

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 30;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

const INITIAL_FLEX: f64 = 0.30;

const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KP_ANG: f64 =  5_000.0;

const ELBOW_DAMPING:   f64 = 0.05;
const ELBOW_STIFFNESS: f64 = 1.5;
const ELBOW_NEUTRAL:   f64 = 0.20;

const ELBOW_LIMIT_MIN: f64 = -0.40;
const ELBOW_LIMIT_MAX: f64 =  2.20;
const ELBOW_LIMIT_STIFFNESS: f64 = 500.0;
const ELBOW_LIMIT_EXPONENT:  f64 =   2.0;

const CTRL_KP: f64 = 30.0;
const CTRL_KD: f64 =  2.0;

const UPPER_REST_Y: f64 = 1.33;
const ELBOW_WORLD_Y: f64 = UPPER_REST_Y - 0.15;

const SCHEDULE_PERIOD: f64 = 7.0;

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    upper_idx: usize,
    lower_idx: usize,
    upper_rest_pos: NaVec3,
    sim_time: f64,
}

#[derive(Component)]
struct BodyRef(usize);

#[derive(Component)]
struct TargetReadout;

fn target_at(t: f64) -> f64 {
    let t = t.rem_euclid(SCHEDULE_PERIOD);
    if      t < 1.0 { 0.30 }
    else if t < 2.5 { 0.50 }
    else if t < 4.0 { 1.00 }
    else if t < 5.5 { 0.30 }
    else            { 0.00 }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();

    // Bodies
    let upper_def = segment_def(BodyType::Receiver, SegmentId::UpperArmL);
    let mut upper = RigidBody::new(upper_def.mass, upper_def.inertia_tensor());
    upper.position = NaVec3::new(0.0, UPPER_REST_Y, 0.0);
    let upper_idx = world.add_body(upper);

    let lower_def = segment_def(BodyType::Receiver, SegmentId::LowerArmL);
    let mut lower = RigidBody::new(lower_def.mass, lower_def.inertia_tensor());
    lower.orientation = UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, INITIAL_FLEX));
    let rotated_anchor = lower.orientation * NaVec3::new(0.0, 0.13, 0.0);
    lower.position = NaVec3::new(0.0, ELBOW_WORLD_Y, 0.0) - rotated_anchor;
    let lower_idx = world.add_body(lower);

    // Joint primitives
    let mut elbow_pos = PositionalConstraint::new(upper_idx, lower_idx, 0.0, 0.0);
    elbow_pos.local_anchor_a = NaVec3::new(0.0, -0.15, 0.0);
    elbow_pos.local_anchor_b = NaVec3::new(0.0,  0.13, 0.0);
    world.constraints.push(elbow_pos);

    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, NaVec3::x(), -0.05, 0.05));
    world.angular_constraints.push(
        AngularLimitConstraint::new(upper_idx, lower_idx, NaVec3::y(), -0.05, 0.05));
    world.angular_soft_limits.push(AngularSoftLimit::new(
        upper_idx, lower_idx, NaVec3::z(),
        ELBOW_LIMIT_MIN, ELBOW_LIMIT_MAX,
        ELBOW_LIMIT_STIFFNESS, ELBOW_LIMIT_EXPONENT,
    ));
    world.angular_dampers.push(
        AngularDamper::new(upper_idx, lower_idx, NaVec3::z(), ELBOW_DAMPING));
    world.angular_springs.push(
        AngularSpring::new(upper_idx, lower_idx, NaVec3::z(), ELBOW_STIFFNESS, ELBOW_NEUTRAL));
    world.angular_pd_controllers.push(
        AngularPDController::new(upper_idx, lower_idx, NaVec3::z(), CTRL_KP, CTRL_KD)
            .with_target(INITIAL_FLEX));

    let phys = PhysicsWorld {
        world,
        upper_idx,
        lower_idx,
        upper_rest_pos: NaVec3::new(0.0, UPPER_REST_Y, 0.0),
        sim_time: 0.0,
    };

    // Capsule meshes — Bevy's Capsule3d::new(radius, length_of_cylindrical_section).
    // segment_def's half_height is half the cylinder length, so length = 2 * half_height.
    let upper_mesh = meshes.add(Capsule3d::new(
        upper_def.radius as f32,
        (2.0 * upper_def.half_height) as f32,
    ));
    let lower_mesh = meshes.add(Capsule3d::new(
        lower_def.radius as f32,
        (2.0 * lower_def.half_height) as f32,
    ));

    let upper_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.78, 0.58, 0.45),
        ..default()
    });
    let lower_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.65, 0.50),
        ..default()
    });

    commands.spawn((
        PbrBundle {
            mesh: upper_mesh,
            material: upper_material,
            transform: rigid_body_to_transform(&phys.world.bodies[upper_idx]),
            ..default()
        },
        BodyRef(upper_idx),
    ));
    commands.spawn((
        PbrBundle {
            mesh: lower_mesh,
            material: lower_material,
            transform: rigid_body_to_transform(&phys.world.bodies[lower_idx]),
            ..default()
        },
        BodyRef(lower_idx),
    ));

    // Small sphere at the shoulder anchor for visual reference.
    let shoulder_mesh = meshes.add(Sphere::new(0.025));
    let shoulder_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.3, 0.3),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: shoulder_mesh,
        material: shoulder_material,
        transform: Transform::from_xyz(0.0, UPPER_REST_Y as f32 + 0.15, 0.0),
        ..default()
    });

    // Ground plane for visual reference.
    let ground_mesh = meshes.add(Plane3d::default().mesh().size(4.0, 4.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: ground_mesh,
        material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });

    // Camera — slight side-front angle, looking at the elbow.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.9, 1.2, 1.4)
            .looking_at(Vec3::new(0.0, 1.05, 0.0), Vec3::Y),
        ..default()
    });

    // Sun
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10_000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(2.0, 4.0, 2.0)
            .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        ..default()
    });
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 220.0,
    });

    // Target readout in the corner.
    commands.spawn((
        TextBundle::from_section(
            "target: 0.30 rad",
            TextStyle {
                font_size: 22.0,
                color: Color::WHITE,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            ..default()
        }),
        TargetReadout,
    ));

    commands.insert_resource(phys);
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let target = target_at(phys.sim_time);
    phys.world.angular_pd_controllers[0].target_angle = target;

    let upper_rest = phys.upper_rest_pos;
    let upper_idx = phys.upper_idx;
    for _ in 0..SOLVER_ITERATIONS {
        let body = &mut phys.world.bodies[upper_idx];
        let pos_err = body.position - upper_rest;
        let force = -ANCHOR_KP_LIN * pos_err;
        let orient_err = body.orientation.scaled_axis();
        let torque = -ANCHOR_KP_ANG * orient_err;
        body.apply_force(force);
        body.apply_torque(torque);

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
    mut query: Query<&mut Text, With<TargetReadout>>,
) {
    let target = target_at(phys.sim_time);
    let body = &phys.world.bodies[phys.lower_idx];
    let upper = &phys.world.bodies[phys.upper_idx];
    let elbow_z = (body.orientation * upper.orientation.inverse()).scaled_axis().z;
    for mut text in &mut query {
        text.sections[0].value = format!(
            "target: {:.2} rad ({:.1}°)\nactual: {:.3} rad ({:.1}°)\nt: {:.2} s",
            target, target.to_degrees(),
            elbow_z, elbow_z.to_degrees(),
            phys.sim_time,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Elbow Demo".into(),
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
