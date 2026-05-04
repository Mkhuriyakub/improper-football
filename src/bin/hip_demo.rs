// Visual demo of the hip covering its full anatomical range of motion.
//
// Anatomical axis mapping for a right hip with the pelvis at identity
// orientation (chest faces +z toward camera, head up +y, right thigh
// hanging down -y from a hip anchor on the +x side):
//
//   FLEX/EXT  →  rotation about world x.  Sagittal plane motion (knee
//                forward toward chest, or thigh extending behind body).
//                axis_local = (-1, 0, 0) so positive biomech angle
//                corresponds to FLEXION (forward). Range: -0.35 (ext)
//                to +2.09 (knee-to-chest).
//
//   ABD/ADD   →  rotation about world z.  Frontal plane motion (leg
//                lifted out to side, or pulled across midline). axis_local
//                = (0, 0, 1) so positive biomech is ABDUCTION (lateral).
//                Range: -0.52 (add) to +0.78 (abd).
//
//   INT/EXT   →  rotation about world y.  Axial twist of femoral shaft.
//                axis_local = (0, 1, 0). Sign convention: positive =
//                EXTERNAL rotation. Range: -0.61 (int) to +0.87 (ext).
//
// Schedule (56 s, looping). Each phase uses a half-cosine ramp so velocity
// is zero at the endpoints:
//
//   0–6 s     forward flexion             FLEX  0 → +1.80 → 0  (knee to chest)
//   6–10 s    backward extension          FLEX  0 → -0.30 → 0
//   10–16 s   lateral abduction           ABD   0 → +0.70 → 0
//   16–20 s   cross-body adduction        ABD   0 → -0.45 → 0
//   20–24 s   internal rotation           ROT   0 → -0.55 → 0
//   24–28 s   external rotation           ROT   0 → +0.80 → 0
//   28–36 s   kick wind-up pose           FLEX → +1.40 AND ABD → +0.30
//   36–44 s   plant-and-twist pose        FLEX → +0.40 AND ABD → -0.40
//   44–56 s   hip circumduction           FLEX/ABD circle, two laps

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
use improper_football::render_util::{rigid_body_to_transform, nalgebra_vec3_to_bevy, nalgebra_quat_to_bevy};

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

const PELVIS_HALF_H: f64 = 0.10;
const PELVIS_RADIUS: f64 = 0.18;

const THIGH_MASS:   f64 = 7.0;
const THIGH_RADIUS: f64 = 0.070;
const THIGH_HALF_H: f64 = 0.225;

const HIP_JOINT_K: f64 = 200_000.0;
const HIP_JOINT_C: f64 =     400.0;

const FLEX_K: f64 = 10.0;
const ABD_K:  f64 = 10.0;
const ROT_K:  f64 =  3.0;
const FLEX_C: f64 = 7.0;
const ABD_C:  f64 = 7.0;
const ROT_C:  f64 = 0.40;
const FLEX_MIN: f64 = -0.35;
const FLEX_MAX: f64 =  2.09;
const ABD_MIN:  f64 = -0.52;
const ABD_MAX:  f64 =  0.78;
const ROT_MIN:  f64 = -0.61;
const ROT_MAX:  f64 =  0.87;
const HIP_LIMIT_K:   f64 = 120.0;
const HIP_LIMIT_EXP: f64 =   2.0;

// Active drive — KP/KD scale up from shoulder values (200/8 for 2 kg arm)
// to match the heavier thigh (7 kg, I_at_hinge ≈ 0.48). KP=500 gives
// ω_n ≈ 32 rad/s for the hinge equation; KD=30 → ζ ≈ 0.96 (just under
// critical, fast tracking with no overshoot).
const HIP_PD_KP: f64 = 500.0;
const HIP_PD_KD: f64 =  30.0;

const PELVIS_REST_POS: NaVec3 = NaVec3::new(0.0, 1.00, 0.0);
const HIP_ANCHOR_PELVIS_X: f64 =  0.10;
const HIP_ANCHOR_PELVIS_Y: f64 = -0.05;

const SCHEDULE_PERIOD: f64 = 56.0;

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

fn add_hip_axis(
    world: &mut XpbdWorld,
    parent: usize, child: usize,
    axis_local_a: NaVec3,
    min_angle: f64, max_angle: f64,
    stiffness: f64, damping: f64,
) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, axis_local_a,
        min_angle, max_angle, HIP_LIMIT_K, HIP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis_local_a, damping));
    world.angular_springs.push(AngularSpring::new(parent, child, axis_local_a, stiffness, 0.0));
}

fn half_cosine(t: f64, duration: f64, peak: f64) -> f64 {
    let phase = (t / duration).clamp(0.0, 1.0);
    peak * 0.5 * (1.0 - (std::f64::consts::TAU * phase).cos())
}

/// (flex_target, abd_target, rot_target, phase_label).
fn hip_targets(t: f64) -> (f64, f64, f64, &'static str) {
    let t = t.rem_euclid(SCHEDULE_PERIOD);
    if t < 6.0 {
        (half_cosine(t, 6.0, 1.80), 0.0, 0.0, "forward flexion (knee to chest)")
    } else if t < 10.0 {
        (half_cosine(t - 6.0, 4.0, -0.30), 0.0, 0.0, "extension (back kick)")
    } else if t < 16.0 {
        (0.0, half_cosine(t - 10.0, 6.0, 0.70), 0.0, "lateral abduction")
    } else if t < 20.0 {
        (0.0, half_cosine(t - 16.0, 4.0, -0.45), 0.0, "cross-body adduction")
    } else if t < 24.0 {
        (0.0, 0.0, half_cosine(t - 20.0, 4.0, -0.55), "internal rotation")
    } else if t < 28.0 {
        (0.0, 0.0, half_cosine(t - 24.0, 4.0, 0.80), "external rotation")
    } else if t < 36.0 {
        // Kick wind-up: knee lifted forward AND slight abduction.
        let local = t - 28.0;
        let flex = half_cosine(local, 8.0, 1.40);
        let abd  = half_cosine(local, 8.0, 0.30);
        (flex, abd, 0.0, "kick wind-up")
    } else if t < 44.0 {
        // Plant-and-twist: slight flex + cross-body adduction (planting
        // weight on the leg with a slight medial tilt).
        let local = t - 36.0;
        let flex = half_cosine(local, 8.0, 0.40);
        let abd  = half_cosine(local, 8.0, -0.40);
        (flex, abd, 0.0, "plant-and-twist")
    } else {
        // Two laps of hip circumduction. Center of circle in front-and-side
        // of body; amplitude shrunken so we don't punch the soft limits.
        let local = (t - 44.0).clamp(0.0, 12.0);
        let envelope = (std::f64::consts::PI * local / 12.0).sin();
        let phase = std::f64::consts::TAU * 2.0 * local / 12.0;
        let center_flex = 0.80;
        let center_abd  = 0.20;
        let amp_flex = 0.80;
        let amp_abd  = 0.45;
        (center_flex * envelope + amp_flex * envelope * phase.sin(),
         center_abd  * envelope + amp_abd  * envelope * phase.cos(),
         0.0,
         "circumduction")
    }
}

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    pelvis_idx: usize,
    thigh_idx: usize,
    sim_time: f64,
}

#[derive(Component)]
struct BodyRef(usize);

#[derive(Component)]
struct KneeMarker;

#[derive(Component)]
struct Readout;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();

    let mut pelvis = RigidBody::new(0.0, capsule_inertia(15.0, PELVIS_RADIUS, PELVIS_HALF_H));
    pelvis.position = PELVIS_REST_POS;
    pelvis.orientation = UnitQuaternion::identity();
    let pelvis_idx = world.add_body(pelvis);

    let mut thigh = RigidBody::new(THIGH_MASS,
        capsule_inertia(THIGH_MASS, THIGH_RADIUS, THIGH_HALF_H));
    let hip_anchor_pelvis = NaVec3::new(HIP_ANCHOR_PELVIS_X, HIP_ANCHOR_PELVIS_Y, 0.0);
    let (pos, orient) = place_child(
        &world.bodies[pelvis_idx],
        hip_anchor_pelvis,
        NaVec3::new(0.0, THIGH_HALF_H, 0.0),
        UnitQuaternion::identity());
    thigh.position = pos;
    thigh.orientation = orient;
    let thigh_idx = world.add_body(thigh);

    add_elastic_joint(&mut world, pelvis_idx, thigh_idx,
        hip_anchor_pelvis,
        NaVec3::new(0.0, THIGH_HALF_H, 0.0),
        HIP_JOINT_K, HIP_JOINT_C);

    // Anatomical biomech axes: same convention as shoulder_demo.
    //   flex about -x (so + biomech = forward flex)
    //   abduction about +z (so + biomech = lateral abd)
    //   rotation about +y (so + biomech = external rotation)
    let flex_axis = NaVec3::new(-1.0, 0.0, 0.0);
    let abd_axis  = NaVec3::new( 0.0, 0.0, 1.0);
    let rot_axis  = NaVec3::new( 0.0, 1.0, 0.0);

    add_hip_axis(&mut world, pelvis_idx, thigh_idx, flex_axis, FLEX_MIN, FLEX_MAX, FLEX_K, FLEX_C);
    add_hip_axis(&mut world, pelvis_idx, thigh_idx, abd_axis,  ABD_MIN,  ABD_MAX,  ABD_K,  ABD_C);
    add_hip_axis(&mut world, pelvis_idx, thigh_idx, rot_axis,  ROT_MIN,  ROT_MAX,  ROT_K,  ROT_C);

    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thigh_idx, flex_axis, HIP_PD_KP, HIP_PD_KD)
            .with_target(0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thigh_idx, abd_axis, HIP_PD_KP, HIP_PD_KD)
            .with_target(0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thigh_idx, rot_axis, HIP_PD_KP, HIP_PD_KD)
            .with_target(0.0));

    let pelvis_mesh = meshes.add(Capsule3d::new(PELVIS_RADIUS as f32, (2.0 * PELVIS_HALF_H) as f32));
    let thigh_mesh  = meshes.add(Capsule3d::new(THIGH_RADIUS as f32, (2.0 * THIGH_HALF_H) as f32));

    let pelvis_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.55, 0.62), ..default() });
    let thigh_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.65, 0.50), ..default() });

    commands.spawn((PbrBundle {
        mesh: pelvis_mesh, material: pelvis_mat,
        transform: rigid_body_to_transform(&world.bodies[pelvis_idx]),
        ..default()
    }, BodyRef(pelvis_idx)));
    commands.spawn((PbrBundle {
        mesh: thigh_mesh, material: thigh_mat,
        transform: rigid_body_to_transform(&world.bodies[thigh_idx]),
        ..default()
    }, BodyRef(thigh_idx)));

    // Knee marker — perpendicular capsule at the thigh's distal end. Same
    // role as the elbow marker on the shoulder demo: makes axial rotation
    // (Y biomech) visible (the thigh spinning around its own long axis is
    // otherwise invisible).
    let marker_mesh = meshes.add(Capsule3d::new(0.014, 0.08));
    let marker_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.30, 0.40, 0.65), ..default() });
    commands.spawn((PbrBundle {
        mesh: marker_mesh, material: marker_mat,
        transform: Transform::default(),
        ..default()
    }, KneeMarker));

    let ground_mesh = meshes.add(Plane3d::default().mesh().size(3.0, 3.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });

    // Camera: 3/4 view from upper-front-right. Frame the hip joint and the
    // full ROM swing of the thigh (which can reach forward to ~-z and out
    // sideways to ~+x).
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(1.20, 1.20, 1.30)
            .looking_at(Vec3::new(0.10, 0.80, 0.0), Vec3::Y),
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
        world, pelvis_idx, thigh_idx, sim_time: 0.0,
    });
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let (flex_t, abd_t, rot_t, _) = hip_targets(phys.sim_time);
    phys.world.angular_pd_controllers[0].target_angle = flex_t;
    phys.world.angular_pd_controllers[1].target_angle = abd_t;
    phys.world.angular_pd_controllers[2].target_angle = rot_t;
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

/// Position the knee-marker capsule perpendicular to the thigh at its
/// distal end. Long axis points along thigh-local +x so the marker rotates
/// visibly when the Y biomech axis (axial twist) is driven.
fn sync_knee_marker(
    phys: Res<PhysicsWorld>,
    mut query: Query<&mut Transform, With<KneeMarker>>,
) {
    let thigh = &phys.world.bodies[phys.thigh_idx];
    let local_offset = NaVec3::new(0.08, -THIGH_HALF_H, 0.0);
    let world_pos = thigh.position + thigh.orientation * local_offset;
    let marker_rot = thigh.orientation
        * UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, -std::f64::consts::FRAC_PI_2));
    for mut transform in &mut query {
        *transform = Transform {
            translation: nalgebra_vec3_to_bevy(world_pos),
            rotation: nalgebra_quat_to_bevy(marker_rot),
            scale: Vec3::ONE,
        };
    }
}

fn update_readout(
    phys: Res<PhysicsWorld>,
    mut query: Query<&mut Text, With<Readout>>,
) {
    let pelvis = &phys.world.bodies[phys.pelvis_idx];
    let thigh = &phys.world.bodies[phys.thigh_idx];
    let q_rel = thigh.orientation * pelvis.orientation.inverse();
    let s = q_rel.scaled_axis();
    let flex_actual = s.dot(&NaVec3::new(-1.0, 0.0, 0.0));
    let abd_actual  = s.dot(&NaVec3::new( 0.0, 0.0, 1.0));
    let rot_actual  = s.dot(&NaVec3::new( 0.0, 1.0, 0.0));
    let (flex_t, abd_t, rot_t, phase) = hip_targets(phys.sim_time);
    for mut text in &mut query {
        text.sections[0].value = format!(
            "{}\nt: {:.2} s  (cycle {:.0} s)\n\n  axis              target → actual\n  flex/ext  : {:+.2}  →  {:+.2}\n  abd/add   : {:+.2}  →  {:+.2}\n  rotation  : {:+.2}  →  {:+.2}",
            phase, phys.sim_time, SCHEDULE_PERIOD,
            flex_t, flex_actual,
            abd_t, abd_actual,
            rot_t, rot_actual,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Hip demo (full ROM, 3-DOF ball)".into(),
                resolution: (1024.0, 768.0).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Time::<Fixed>::from_hz(PHYSICS_HZ))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, physics_step)
        .add_systems(Update, (sync_transforms, sync_knee_marker, update_readout))
        .run();
}
