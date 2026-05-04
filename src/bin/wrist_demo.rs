// Visual demo of the wrist on its own (no fingers yet).
//
// Pinned horizontal forearm stub with a palm on the elastic-joint wrist.
// Full passive biomechanics (Z flex/ext, X radial/ulnar, Y locked) plus
// active AngularPDControllers on Z and X driving the palm slowly through
// the full anatomical range of motion:
//
//   0–6 s   palmar flexion sweep (Z: 0 → +1.20 → 0)
//   6–12 s  dorsi extension sweep (Z: 0 → −1.00 → 0)
//   12–18 s ulnar deviation sweep (X: 0 → +0.55 → 0)
//   18–24 s radial deviation sweep (X: 0 → −0.30 → 0)
//   24–36 s combined Z+X circumduction (slow circle, two laps)
//
// Then loops. Targets are half-cosine ramps so each sweep eases in and out.

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

const FOREARM_HALF_H: f64 = 0.135;
const FOREARM_RADIUS: f64 = 0.040;

const PALM_MASS:    f64 = 0.30;
const PALM_HALF_H:  f64 = 0.040;
const PALM_RADIUS:  f64 = 0.030;

const JOINT_K: f64 = 50_000.0;
const JOINT_C: f64 =    100.0;

// Passive: gentler springs so the active drive can take the wrist near its
// anatomical limits without fighting the spring; bumped damping ratios so
// any residual oscillation settles within one cycle.
const WRIST_Z_K:        f64 = 0.4;
const WRIST_X_K:        f64 = 0.6;
// Y axis: anatomically there's no DOF here (axial twist happens at the
// radioulnar joints, not the wrist). But our angular primitives use
// scaled_axis().dot(axis), which mixes in a bivector artifact when Z and X
// are both rotated — at Z=0.9, X=0.4 the apparent y angle reads ~0.18 even
// though no real twist occurred. We relax the y bounds so the limit doesn't
// fight that artifact, and keep a soft spring to pull back to zero between
// motions.
const WRIST_Y_K:        f64 = 0.5;
// Y damper is critical: the bivector artifact pumps energy into the y mode
// every step of the circumduction phase. Without damping, that mode rings at
// its natural frequency (~10 Hz) for the duration, which reads as jitter.
const WRIST_Y_C:        f64 = 0.015;
const WRIST_Z_C:        f64 = 0.05;
const WRIST_X_C:        f64 = 0.06;
const WRIST_Z_MIN:      f64 = -1.10;
const WRIST_Z_MAX:      f64 =  1.30;
const WRIST_X_MIN:      f64 = -0.35;
const WRIST_X_MAX:      f64 =  0.60;
const WRIST_Y_LIM:      f64 =  0.30;
const WRIST_LIMIT_K:    f64 = 200.0;
const WRIST_Y_LIMIT_K:  f64 =  20.0;
const WRIST_LIMIT_EXP:  f64 =   2.0;

// Active drive (PD on top of passive biomech).
const WRIST_PD_KP: f64 = 50.0;
const WRIST_PD_KD: f64 =  1.0;

const FOREARM_REST_POS: NaVec3 = NaVec3::new(0.0, 1.0, 0.0);

const SCHEDULE_PERIOD: f64 = 36.0;
const Z_FLEX_PEAK:  f64 =  1.20;
const Z_EXT_PEAK:   f64 = -1.00;
const X_ULNAR_PEAK: f64 =  0.55;
const X_RADIAL_PEAK:f64 = -0.30;
const CIRCLE_AMP_Z: f64 =  0.90;
const CIRCLE_AMP_X: f64 =  0.40;

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

/// Half-cosine ramp from 0 to `peak` and back over `[0, duration]`.
/// At t = 0 and t = duration the value is 0 and the velocity is 0.
fn half_cosine(t: f64, duration: f64, peak: f64) -> f64 {
    let phase = (t / duration).clamp(0.0, 1.0);
    peak * 0.5 * (1.0 - (std::f64::consts::TAU * phase).cos())
}

fn targets_at(t: f64) -> (f64, f64, &'static str) {
    let t = t.rem_euclid(SCHEDULE_PERIOD);
    if t < 6.0 {
        (half_cosine(t, 6.0, Z_FLEX_PEAK), 0.0, "palmar flexion (Z)")
    } else if t < 12.0 {
        (half_cosine(t - 6.0, 6.0, Z_EXT_PEAK), 0.0, "dorsi extension (Z)")
    } else if t < 18.0 {
        (0.0, half_cosine(t - 12.0, 6.0, X_ULNAR_PEAK), "ulnar deviation (X)")
    } else if t < 24.0 {
        (0.0, half_cosine(t - 18.0, 6.0, X_RADIAL_PEAK), "radial deviation (X)")
    } else {
        // Two slow laps of circumduction over 12 s. Amplitude is enveloped by
        // a half-sine so the circle ramps in from zero and back to zero,
        // avoiding the step that would otherwise hit the X target at t=24.
        let local = (t - 24.0).clamp(0.0, 12.0);
        let envelope = (std::f64::consts::PI * local / 12.0).sin();
        let phase = std::f64::consts::TAU * 2.0 * local / 12.0;
        (CIRCLE_AMP_Z * envelope * phase.sin(),
         CIRCLE_AMP_X * envelope * phase.cos(),
         "circumduction")
    }
}

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    forearm_idx: usize,
    palm_idx: usize,
    sim_time: f64,
}

#[derive(Component)]
struct BodyRef(usize);

#[derive(Component)]
struct AngleReadout;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();

    let mut forearm = RigidBody::new(
        0.0,
        capsule_inertia(1.5, FOREARM_RADIUS, FOREARM_HALF_H),
    );
    forearm.position = FOREARM_REST_POS;
    forearm.orientation = UnitQuaternion::from_scaled_axis(
        NaVec3::new(0.0, 0.0, -std::f64::consts::FRAC_PI_2),
    );
    let forearm_idx = world.add_body(forearm);

    let mut palm = RigidBody::new(
        PALM_MASS,
        capsule_inertia(PALM_MASS, PALM_RADIUS, PALM_HALF_H),
    );
    let (pos, orient) = place_child(
        &world.bodies[forearm_idx],
        NaVec3::new(0.0, -FOREARM_HALF_H, 0.0),
        NaVec3::new(0.0,  PALM_HALF_H, 0.0),
        UnitQuaternion::identity(),
    );
    palm.position = pos;
    palm.orientation = orient;
    let palm_idx = world.add_body(palm);

    let mut spring = PositionalSpring::new(forearm_idx, palm_idx, JOINT_K);
    spring.local_anchor_a = NaVec3::new(0.0, -FOREARM_HALF_H, 0.0);
    spring.local_anchor_b = NaVec3::new(0.0,  PALM_HALF_H,    0.0);
    world.positional_springs.push(spring);

    let mut damp = PositionalDamper::new(forearm_idx, palm_idx, JOINT_C);
    damp.local_anchor_a = NaVec3::new(0.0, -FOREARM_HALF_H, 0.0);
    damp.local_anchor_b = NaVec3::new(0.0,  PALM_HALF_H,    0.0);
    world.positional_dampers.push(damp);

    world.angular_soft_limits.push(AngularSoftLimit::new(
        forearm_idx, palm_idx, NaVec3::z(),
        WRIST_Z_MIN, WRIST_Z_MAX, WRIST_LIMIT_K, WRIST_LIMIT_EXP));
    world.angular_dampers.push(
        AngularDamper::new(forearm_idx, palm_idx, NaVec3::z(), WRIST_Z_C));
    world.angular_springs.push(
        AngularSpring::new(forearm_idx, palm_idx, NaVec3::z(), WRIST_Z_K, 0.0));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        forearm_idx, palm_idx, NaVec3::x(),
        WRIST_X_MIN, WRIST_X_MAX, WRIST_LIMIT_K, WRIST_LIMIT_EXP));
    world.angular_dampers.push(
        AngularDamper::new(forearm_idx, palm_idx, NaVec3::x(), WRIST_X_C));
    world.angular_springs.push(
        AngularSpring::new(forearm_idx, palm_idx, NaVec3::x(), WRIST_X_K, 0.0));

    world.angular_soft_limits.push(AngularSoftLimit::new(
        forearm_idx, palm_idx, NaVec3::y(),
        -WRIST_Y_LIM, WRIST_Y_LIM, WRIST_Y_LIMIT_K, WRIST_LIMIT_EXP));
    world.angular_dampers.push(
        AngularDamper::new(forearm_idx, palm_idx, NaVec3::y(), WRIST_Y_C));
    world.angular_springs.push(
        AngularSpring::new(forearm_idx, palm_idx, NaVec3::y(), WRIST_Y_K, 0.0));

    // Active drive: PD on Z and X axes. Y is locked, no controller needed.
    world.angular_pd_controllers.push(
        AngularPDController::new(forearm_idx, palm_idx, NaVec3::z(), WRIST_PD_KP, WRIST_PD_KD)
            .with_target(0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(forearm_idx, palm_idx, NaVec3::x(), WRIST_PD_KP, WRIST_PD_KD)
            .with_target(0.0));

    let phys = PhysicsWorld {
        world,
        forearm_idx,
        palm_idx,
        sim_time: 0.0,
    };

    let forearm_mesh = meshes.add(Capsule3d::new(
        FOREARM_RADIUS as f32, (2.0 * FOREARM_HALF_H) as f32));
    let palm_mesh = meshes.add(Capsule3d::new(
        PALM_RADIUS as f32, (2.0 * PALM_HALF_H) as f32));

    let forearm_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.78, 0.58, 0.45),
        ..default()
    });
    let palm_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.92, 0.72, 0.58),
        ..default()
    });

    commands.spawn((
        PbrBundle {
            mesh: forearm_mesh, material: forearm_mat,
            transform: rigid_body_to_transform(&phys.world.bodies[forearm_idx]),
            ..default()
        },
        BodyRef(forearm_idx),
    ));
    commands.spawn((
        PbrBundle {
            mesh: palm_mesh, material: palm_mat,
            transform: rigid_body_to_transform(&phys.world.bodies[palm_idx]),
            ..default()
        },
        BodyRef(palm_idx),
    ));

    let ground_mesh = meshes.add(Plane3d::default().mesh().size(2.0, 2.0));
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

    // Camera placed slightly forward and above, looking at the wrist joint
    // from a 3/4 angle so both Z (flex/ext) and X (rad/uln) motions read.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.4, 1.15, 0.55)
            .looking_at(Vec3::new(-0.18, 0.92, 0.0), Vec3::Y),
        ..default()
    });

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

    commands.spawn((
        TextBundle::from_section(
            "...",
            TextStyle {
                font_size: 20.0,
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
        AngleReadout,
    ));

    commands.insert_resource(phys);
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let (z_target, x_target, _) = targets_at(phys.sim_time);
    phys.world.angular_pd_controllers[0].target_angle = z_target;
    phys.world.angular_pd_controllers[1].target_angle = x_target;
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
    mut query: Query<&mut Text, With<AngleReadout>>,
) {
    let f = &phys.world.bodies[phys.forearm_idx];
    let p = &phys.world.bodies[phys.palm_idx];
    let rel = (p.orientation * f.orientation.inverse()).scaled_axis();
    let (z_t, x_t, phase) = targets_at(phys.sim_time);
    for mut text in &mut query {
        text.sections[0].value = format!(
            "{}\nt: {:.2} s  (cycle {:.0} s)\n\n  axis      target → actual\n  flex (Z): {:+.2} → {:+.2} rad\n  dev  (X): {:+.2} → {:+.2} rad\n  twist(Y):           {:+.3} rad",
            phase, phys.sim_time, SCHEDULE_PERIOD,
            z_t, rel.z,
            x_t, rel.x,
            rel.y,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Wrist demo (full ROM)".into(),
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
