// Visual demo of the shoulder covering the full anatomical range of motion.
//
// Anatomical axis mapping for a right shoulder with the torso at identity
// orientation (chest faces +z toward camera, head up +y, right arm hanging
// down -y from a shoulder anchor on the +x side):
//
//   FLEX/EXT  →  rotation about world x.  Sagittal plane motion (arm forward
//                or back). axis_local = (-1, 0, 0) so positive biomech angle
//                corresponds to FLEXION (forward) — the larger anatomical
//                range. Range: -0.87 (ext) to +3.14 (overhead flex).
//
//   ABD/ADD   →  rotation about world z.  Frontal plane motion (arm out to
//                side or across body). axis_local = (0, 0, 1) so positive
//                biomech angle is ABDUCTION (lateral) — again the larger
//                range. Range: -0.78 (add) to +3.14 (overhead abd).
//
//   INT/EXT   →  rotation about world y.  Transverse plane motion (humerus
//                axial twist). axis_local = (0, 1, 0). Sign convention:
//                positive = EXTERNAL rotation. Range: -1.40 (int) to +1.57.
//
// Schedule (56 s, looping). Each phase uses a half-cosine ramp so velocity
// is zero at the endpoints. Only one or two axes are active at a time;
// idle axes are commanded to 0 and held there.
//
//   0–6 s     forward flexion             FLEX  0 → +3.00 → 0  (full overhead)
//   6–10 s    backward extension          FLEX  0 → -0.80 → 0
//   10–16 s   lateral abduction           ABD   0 → +3.00 → 0  (full overhead)
//   16–20 s   cross-body adduction        ABD   0 → -0.70 → 0
//   20–24 s   internal rotation           ROT   0 → -1.30 → 0
//   24–28 s   external rotation           ROT   0 → +1.50 → 0
//   28–36 s   throwing cock pose          ABD → +1.57 AND ROT → +1.40
//   36–44 s   cross-body reach            FLEX → +1.00 AND ABD → -0.40
//   44–56 s   forward circumduction       FLEX/ABD circle, two laps

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

const TORSO_HALF_H: f64 = 0.20;
const TORSO_RADIUS: f64 = 0.15;

const UPPER_ARM_MASS:    f64 = 2.0;
const UPPER_ARM_RADIUS:  f64 = 0.045;
const UPPER_ARM_HALF_H:  f64 = 0.150;

const SHOULDER_JOINT_K: f64 = 100_000.0;
const SHOULDER_JOINT_C: f64 =     200.0;

const FLEX_K: f64 = 3.0;
const ABD_K:  f64 = 3.0;
const ROT_K:  f64 = 1.5;
const FLEX_C: f64 = 0.30;
const ABD_C:  f64 = 0.30;
const ROT_C:  f64 = 0.10;
const FLEX_MIN: f64 = -0.87;
const FLEX_MAX: f64 =  3.14;
const ABD_MIN:  f64 = -0.78;
const ABD_MAX:  f64 =  3.14;
const ROT_MIN:  f64 = -1.40;
const ROT_MAX:  f64 =  1.57;
const SHOULDER_LIMIT_K:   f64 = 100.0;
const SHOULDER_LIMIT_EXP: f64 =   2.0;

const SHOULDER_PD_KP: f64 = 200.0;
const SHOULDER_PD_KD: f64 =   8.0;

const TORSO_REST_POS: NaVec3 = NaVec3::new(0.0, 1.40, 0.0);
const SHOULDER_ANCHOR_TORSO_X: f64 = 0.15;
const SHOULDER_ANCHOR_TORSO_Y: f64 = 0.18;

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

/// Add per-axis spring + damper + soft limit on `axis_local_a` (in body_a's
/// frame), with the given anatomical range and stiffness/damping. Used three
/// times for the shoulder (flex/abd/rot, each on a different axis).
fn add_shoulder_axis(
    world: &mut XpbdWorld,
    parent: usize, child: usize,
    axis_local_a: NaVec3,
    min_angle: f64, max_angle: f64,
    stiffness: f64, damping: f64,
) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, axis_local_a,
        min_angle, max_angle, SHOULDER_LIMIT_K, SHOULDER_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis_local_a, damping));
    world.angular_springs.push(AngularSpring::new(parent, child, axis_local_a, stiffness, 0.0));
}

fn half_cosine(t: f64, duration: f64, peak: f64) -> f64 {
    let phase = (t / duration).clamp(0.0, 1.0);
    peak * 0.5 * (1.0 - (std::f64::consts::TAU * phase).cos())
}

/// (flex_target, abd_target, rot_target, phase_label).
fn shoulder_targets(t: f64) -> (f64, f64, f64, &'static str) {
    let t = t.rem_euclid(SCHEDULE_PERIOD);
    if t < 6.0 {
        (half_cosine(t, 6.0, 3.00), 0.0, 0.0, "forward flexion")
    } else if t < 10.0 {
        (half_cosine(t - 6.0, 4.0, -0.80), 0.0, 0.0, "extension (back)")
    } else if t < 16.0 {
        (0.0, half_cosine(t - 10.0, 6.0, 3.00), 0.0, "lateral abduction")
    } else if t < 20.0 {
        (0.0, half_cosine(t - 16.0, 4.0, -0.70), 0.0, "cross-body adduction")
    } else if t < 24.0 {
        (0.0, 0.0, half_cosine(t - 20.0, 4.0, -1.30), "internal rotation")
    } else if t < 28.0 {
        (0.0, 0.0, half_cosine(t - 24.0, 4.0, 1.50), "external rotation")
    } else if t < 36.0 {
        // Throwing cock pose: 90° abduction + max external rotation.
        // Both ramp in over 4 s, hold 0 s, ramp out over 4 s using half cosines.
        let local = t - 28.0;
        let abd = half_cosine(local, 8.0, 1.57);
        let rot = half_cosine(local, 8.0, 1.40);
        (0.0, abd, rot, "throwing cock pose")
    } else if t < 44.0 {
        // Cross-body reach: forward flex + slight cross-body adduction.
        let local = t - 36.0;
        let flex = half_cosine(local, 8.0, 1.00);
        let abd  = half_cosine(local, 8.0, -0.40);
        (flex, abd, 0.0, "cross-body reach")
    } else {
        // Two laps of forward circumduction in front of body — flex sweeps
        // up while abd sweeps to side, traces a wide circle. Enveloped to
        // start/end at zero amplitude.
        let local = (t - 44.0).clamp(0.0, 12.0);
        let envelope = (std::f64::consts::PI * local / 12.0).sin();
        let phase = std::f64::consts::TAU * 2.0 * local / 12.0;
        // Center of circle at (FLEX=1.5, ABD=0.8) so the arm traces a cone in
        // front of and to the side of the body. Amplitude shrunken so we
        // don't punch through the soft limits.
        let center_flex = 1.30;
        let center_abd  = 0.80;
        let amp_flex = 1.20;
        let amp_abd  = 0.80;
        (center_flex * envelope + amp_flex * envelope * phase.sin(),
         center_abd  * envelope + amp_abd  * envelope * phase.cos(),
         0.0,
         "circumduction")
    }
}

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    torso_idx: usize,
    upper_idx: usize,
    sim_time: f64,
}

#[derive(Component)]
struct BodyRef(usize);

#[derive(Component)]
struct ElbowMarker;

#[derive(Component)]
struct Readout;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();

    let mut torso = RigidBody::new(0.0, capsule_inertia(50.0, TORSO_RADIUS, TORSO_HALF_H));
    torso.position = TORSO_REST_POS;
    torso.orientation = UnitQuaternion::identity();
    let torso_idx = world.add_body(torso);

    let mut upper = RigidBody::new(UPPER_ARM_MASS,
        capsule_inertia(UPPER_ARM_MASS, UPPER_ARM_RADIUS, UPPER_ARM_HALF_H));
    let shoulder_anchor_torso = NaVec3::new(SHOULDER_ANCHOR_TORSO_X, SHOULDER_ANCHOR_TORSO_Y, 0.0);
    let (pos, orient) = place_child(
        &world.bodies[torso_idx],
        shoulder_anchor_torso,
        NaVec3::new(0.0, UPPER_ARM_HALF_H, 0.0),
        UnitQuaternion::identity());
    upper.position = pos;
    upper.orientation = orient;
    let upper_idx = world.add_body(upper);

    add_elastic_joint(&mut world, torso_idx, upper_idx,
        shoulder_anchor_torso,
        NaVec3::new(0.0, UPPER_ARM_HALF_H, 0.0),
        SHOULDER_JOINT_K, SHOULDER_JOINT_C);

    // Anatomical biomech axes: flex about -x (so + biomech = forward flex),
    // abduction about +z (so + biomech = lateral abd), rotation about +y
    // (so + biomech = external rotation).
    let flex_axis = NaVec3::new(-1.0, 0.0, 0.0);
    let abd_axis  = NaVec3::new( 0.0, 0.0, 1.0);
    let rot_axis  = NaVec3::new( 0.0, 1.0, 0.0);

    add_shoulder_axis(&mut world, torso_idx, upper_idx, flex_axis, FLEX_MIN, FLEX_MAX, FLEX_K, FLEX_C);
    add_shoulder_axis(&mut world, torso_idx, upper_idx, abd_axis,  ABD_MIN,  ABD_MAX,  ABD_K,  ABD_C);
    add_shoulder_axis(&mut world, torso_idx, upper_idx, rot_axis,  ROT_MIN,  ROT_MAX,  ROT_K,  ROT_C);

    // PD controllers on the same three axes (flex/abd/rot order).
    world.angular_pd_controllers.push(
        AngularPDController::new(torso_idx, upper_idx, flex_axis, SHOULDER_PD_KP, SHOULDER_PD_KD)
            .with_target(0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(torso_idx, upper_idx, abd_axis, SHOULDER_PD_KP, SHOULDER_PD_KD)
            .with_target(0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(torso_idx, upper_idx, rot_axis, SHOULDER_PD_KP, SHOULDER_PD_KD)
            .with_target(0.0));

    let torso_mesh = meshes.add(Capsule3d::new(TORSO_RADIUS as f32, (2.0 * TORSO_HALF_H) as f32));
    let upper_mesh = meshes.add(Capsule3d::new(UPPER_ARM_RADIUS as f32, (2.0 * UPPER_ARM_HALF_H) as f32));

    let torso_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.55, 0.62), ..default() });
    let upper_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.65, 0.50), ..default() });

    commands.spawn((PbrBundle {
        mesh: torso_mesh, material: torso_mat,
        transform: rigid_body_to_transform(&world.bodies[torso_idx]),
        ..default()
    }, BodyRef(torso_idx)));
    commands.spawn((PbrBundle {
        mesh: upper_mesh, material: upper_mat,
        transform: rigid_body_to_transform(&world.bodies[upper_idx]),
        ..default()
    }, BodyRef(upper_idx)));

    // Elbow marker — small thin capsule attached perpendicular to the upper
    // arm at its distal end. Visual only (no physics body); transform is
    // computed from the upper arm's pose each frame. Makes the int/ext
    // rotation phases visible (without it, the upper arm spinning around
    // its own axis is invisible).
    let marker_mesh = meshes.add(Capsule3d::new(0.012, 0.06));
    let marker_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.30, 0.40, 0.65), ..default() });
    commands.spawn((PbrBundle {
        mesh: marker_mesh, material: marker_mat,
        transform: Transform::default(),
        ..default()
    }, ElbowMarker));

    let ground_mesh = meshes.add(Plane3d::default().mesh().size(3.0, 3.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });

    // Camera: 3/4 view from upper-right-front. Sees flex (in/out of screen
    // depth), abd (left-right on screen), full overhead range.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(1.10, 1.55, 1.30)
            .looking_at(Vec3::new(0.15, 1.40, 0.0), Vec3::Y),
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
        world, torso_idx, upper_idx, sim_time: 0.0,
    });
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let (flex_t, abd_t, rot_t, _) = shoulder_targets(phys.sim_time);
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

/// Position the elbow-marker capsule perpendicular to the upper arm at its
/// distal end. Marker's local +y is aligned with upper-arm-local +x, so the
/// marker's long axis points laterally relative to the upper arm and rotates
/// with axial twist (Y biomech axis = int/ext rotation).
fn sync_elbow_marker(
    phys: Res<PhysicsWorld>,
    mut query: Query<&mut Transform, With<ElbowMarker>>,
) {
    let upper = &phys.world.bodies[phys.upper_idx];
    // Center of marker offset 0.06 m along upper-arm-local +x from the
    // distal end (which is at upper-arm-local (0, -UPPER_ARM_HALF_H, 0)).
    let local_offset = NaVec3::new(0.06, -UPPER_ARM_HALF_H, 0.0);
    let world_pos = upper.position + upper.orientation * local_offset;
    // Rotate marker so its long axis (+y in mesh) points along upper-arm-local +x.
    let marker_rot = upper.orientation
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
    let torso = &phys.world.bodies[phys.torso_idx];
    let upper = &phys.world.bodies[phys.upper_idx];
    let q_rel = upper.orientation * torso.orientation.inverse();
    let s = q_rel.scaled_axis();
    // Project onto the same axes the biomech uses, so the readout matches
    // the targets directly.
    let flex_actual = s.dot(&NaVec3::new(-1.0, 0.0, 0.0));
    let abd_actual  = s.dot(&NaVec3::new( 0.0, 0.0, 1.0));
    let rot_actual  = s.dot(&NaVec3::new( 0.0, 1.0, 0.0));
    let (flex_t, abd_t, rot_t, phase) = shoulder_targets(phys.sim_time);
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
                title: "Shoulder demo (full ROM, 3-DOF ball)".into(),
                resolution: (1024.0, 768.0).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Time::<Fixed>::from_hz(PHYSICS_HZ))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, physics_step)
        .add_systems(Update, (sync_transforms, sync_elbow_marker, update_readout))
        .run();
}
