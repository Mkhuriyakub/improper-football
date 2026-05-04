// Visual demo of the cervical (neck) joint covering its full anatomical
// ROM. Same axis convention as the lumbar demo, but with much wider
// rotational range (±80°) and a much lighter load (head ≈ 5 kg).
//
// Anatomical axis mapping (thoracic at identity, person facing +z, head
// up +y, right side at +x):
//
//   FLEX/EXT  →  rotation about world +x.  Sagittal plane (chin to chest
//                or look up). Range: -1.05 (-60° "look up") to +0.87
//                (+50° "chin to chest").
//
//   BEND      →  rotation about world -z.  Frontal plane (ear to shoulder).
//                Range: ±0.78 (±45°).
//
//   ROT       →  rotation about world +y.  Transverse plane (look over
//                right or left shoulder). Range: ±1.40 (±80°).
//
// Schedule (40 s, looping):
//
//   0–6 s     forward flexion       FLEX 0 → +0.75 → 0  ("chin to chest")
//   6–10 s    back extension        FLEX 0 → -0.95 → 0  ("look up")
//   10–14 s   right lateral bend    BEND 0 → +0.65 → 0
//   14–18 s   left lateral bend     BEND 0 → -0.65 → 0
//   18–22 s   right rotation        ROT  0 → +1.20 → 0  (look right)
//   22–26 s   left rotation         ROT  0 → -1.20 → 0
//   26–32 s   look up & turn right  EXT + ROT combined
//   32–40 s   head circumduction    flex/bend traced as a circle, 2 laps

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

const THORACIC_HALF_H: f64 = 0.180;
const THORACIC_RADIUS: f64 = 0.130;
const THORACIC_REST_POS: NaVec3 = NaVec3::new(0.0, 1.28, 0.0);
const CERV_ANCHOR_THORACIC_Y: f64 = THORACIC_HALF_H;

const HEAD_MASS:    f64 = 5.0;
const HEAD_RADIUS:  f64 = 0.080;
const HEAD_HALF_H:  f64 = 0.050;

const CERV_JOINT_K: f64 = 80_000.0;
const CERV_JOINT_C: f64 =    200.0;

const FLEX_K: f64 = 5.0;  const FLEX_C: f64 = 0.60;
const BEND_K: f64 = 5.0;  const BEND_C: f64 = 0.60;
const ROT_K:  f64 = 3.0;  const ROT_C:  f64 = 0.50;
const FLEX_MIN: f64 = -1.05;  const FLEX_MAX: f64 = 0.87;
const BEND_MIN: f64 = -0.78;  const BEND_MAX: f64 = 0.78;
const ROT_MIN:  f64 = -1.40;  const ROT_MAX:  f64 = 1.40;
const CERV_LIMIT_K:   f64 = 80.0;
const CERV_LIMIT_EXP: f64 =  2.0;

// Active drive — head I_at_cerv ≈ 0.042 kg·m². KP=60 gives ω_n ≈ 38 rad/s
// (snappy enough to see motion clearly); KD=3 gives ζ ≈ 0.94.
const CERV_PD_KP: f64 = 60.0;
const CERV_PD_KD: f64 =  3.0;

const SCHEDULE_PERIOD: f64 = 40.0;

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

fn add_cerv_axis(
    world: &mut XpbdWorld,
    parent: usize, child: usize,
    axis_local_a: NaVec3,
    min_angle: f64, max_angle: f64,
    stiffness: f64, damping: f64,
) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, axis_local_a,
        min_angle, max_angle, CERV_LIMIT_K, CERV_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis_local_a, damping));
    world.angular_springs.push(AngularSpring::new(parent, child, axis_local_a, stiffness, 0.0));
}

fn half_cosine(t: f64, duration: f64, peak: f64) -> f64 {
    let phase = (t / duration).clamp(0.0, 1.0);
    peak * 0.5 * (1.0 - (std::f64::consts::TAU * phase).cos())
}

/// (flex_target, bend_target, rot_target, phase_label).
fn cerv_targets(t: f64) -> (f64, f64, f64, &'static str) {
    let t = t.rem_euclid(SCHEDULE_PERIOD);
    if t < 6.0 {
        (half_cosine(t, 6.0, 0.75), 0.0, 0.0, "forward flexion (chin to chest)")
    } else if t < 10.0 {
        (half_cosine(t - 6.0, 4.0, -0.95), 0.0, 0.0, "extension (look up)")
    } else if t < 14.0 {
        (0.0, half_cosine(t - 10.0, 4.0, 0.65), 0.0, "right lateral bend")
    } else if t < 18.0 {
        (0.0, half_cosine(t - 14.0, 4.0, -0.65), 0.0, "left lateral bend")
    } else if t < 22.0 {
        (0.0, 0.0, half_cosine(t - 18.0, 4.0, 1.20), "right rotation (look right)")
    } else if t < 26.0 {
        (0.0, 0.0, half_cosine(t - 22.0, 4.0, -1.20), "left rotation")
    } else if t < 32.0 {
        // Combined: look up and to the right.
        let local = t - 26.0;
        let flex = half_cosine(local, 6.0, -0.70);
        let rot  = half_cosine(local, 6.0, 0.90);
        (flex, 0.0, rot, "look up & turn right")
    } else {
        // Two laps of head circumduction (chin tracing a circle).
        let local = (t - 32.0).clamp(0.0, 8.0);
        let envelope = (std::f64::consts::PI * local / 8.0).sin();
        let phase = std::f64::consts::TAU * 2.0 * local / 8.0;
        let amp_flex = 0.55;
        let amp_bend = 0.55;
        (amp_flex * envelope * phase.sin(),
         amp_bend * envelope * phase.cos(),
         0.0,
         "head circumduction")
    }
}

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    thoracic_idx: usize,
    head_idx: usize,
    sim_time: f64,
}

#[derive(Component)]
struct BodyRef(usize);

#[derive(Component)]
struct NoseMarker;

#[derive(Component)]
struct Readout;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();

    let mut thoracic = RigidBody::new(0.0,
        capsule_inertia(25.0, THORACIC_RADIUS, THORACIC_HALF_H));
    thoracic.position = THORACIC_REST_POS;
    thoracic.orientation = UnitQuaternion::identity();
    let thoracic_idx = world.add_body(thoracic);

    let mut head = RigidBody::new(HEAD_MASS,
        capsule_inertia(HEAD_MASS, HEAD_RADIUS, HEAD_HALF_H));
    let cerv_anchor_thoracic = NaVec3::new(0.0, CERV_ANCHOR_THORACIC_Y, 0.0);
    let (pos, orient) = place_child(
        &world.bodies[thoracic_idx],
        cerv_anchor_thoracic,
        NaVec3::new(0.0, -HEAD_HALF_H, 0.0),
        UnitQuaternion::identity());
    head.position = pos;
    head.orientation = orient;
    let head_idx = world.add_body(head);

    add_elastic_joint(&mut world, thoracic_idx, head_idx,
        cerv_anchor_thoracic,
        NaVec3::new(0.0, -HEAD_HALF_H, 0.0),
        CERV_JOINT_K, CERV_JOINT_C);

    let flex_axis = NaVec3::new(1.0, 0.0, 0.0);
    let bend_axis = NaVec3::new(0.0, 0.0, -1.0);
    let rot_axis  = NaVec3::new(0.0, 1.0, 0.0);

    add_cerv_axis(&mut world, thoracic_idx, head_idx, flex_axis,
        FLEX_MIN, FLEX_MAX, FLEX_K, FLEX_C);
    add_cerv_axis(&mut world, thoracic_idx, head_idx, bend_axis,
        BEND_MIN, BEND_MAX, BEND_K, BEND_C);
    add_cerv_axis(&mut world, thoracic_idx, head_idx, rot_axis,
        ROT_MIN, ROT_MAX, ROT_K, ROT_C);

    world.angular_pd_controllers.push(
        AngularPDController::new(thoracic_idx, head_idx, flex_axis,
            CERV_PD_KP, CERV_PD_KD).with_target(0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(thoracic_idx, head_idx, bend_axis,
            CERV_PD_KP, CERV_PD_KD).with_target(0.0));
    world.angular_pd_controllers.push(
        AngularPDController::new(thoracic_idx, head_idx, rot_axis,
            CERV_PD_KP, CERV_PD_KD).with_target(0.0));

    let thoracic_mesh = meshes.add(Capsule3d::new(THORACIC_RADIUS as f32, (2.0 * THORACIC_HALF_H) as f32));
    let head_mesh     = meshes.add(Capsule3d::new(HEAD_RADIUS as f32, (2.0 * HEAD_HALF_H) as f32));

    let thoracic_mat = materials.add(StandardMaterial { base_color: Color::srgb(0.78, 0.62, 0.50), ..default() });
    let head_mat     = materials.add(StandardMaterial { base_color: Color::srgb(0.92, 0.78, 0.65), ..default() });

    commands.spawn((PbrBundle {
        mesh: thoracic_mesh, material: thoracic_mat,
        transform: rigid_body_to_transform(&world.bodies[thoracic_idx]),
        ..default()
    }, BodyRef(thoracic_idx)));
    commands.spawn((PbrBundle {
        mesh: head_mesh, material: head_mat,
        transform: rigid_body_to_transform(&world.bodies[head_idx]),
        ..default()
    }, BodyRef(head_idx)));

    // Nose marker — small forward-pointing capsule on the front of the
    // head. Makes axial rotation (Y) visible and orientation readable.
    let marker_mesh = meshes.add(Capsule3d::new(0.012, 0.05));
    let marker_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.65, 0.30, 0.30), ..default() });
    commands.spawn((PbrBundle {
        mesh: marker_mesh, material: marker_mat,
        transform: Transform::default(),
        ..default()
    }, NoseMarker));

    let ground_mesh = meshes.add(Plane3d::default().mesh().size(3.0, 3.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });

    // Camera: front-quarter view, framing thoracic-to-top-of-head.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.85, 1.55, 1.10)
            .looking_at(Vec3::new(0.0, 1.50, 0.0), Vec3::Y),
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
        world, thoracic_idx, head_idx, sim_time: 0.0,
    });
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let (flex_t, bend_t, rot_t, _) = cerv_targets(phys.sim_time);
    phys.world.angular_pd_controllers[0].target_angle = flex_t;
    phys.world.angular_pd_controllers[1].target_angle = bend_t;
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

/// Position the nose marker on the +z (front) face of the head, mid-height.
/// Marker mesh's +y aligned with head-local +z so it points forward and
/// rotates visibly under axial rotation (Y) and lateral bend (X).
fn sync_nose_marker(
    phys: Res<PhysicsWorld>,
    mut query: Query<&mut Transform, With<NoseMarker>>,
) {
    let head = &phys.world.bodies[phys.head_idx];
    // Pop slightly forward (+z) of the head's surface, centered vertically.
    let local_offset = NaVec3::new(0.0, 0.0, HEAD_RADIUS + 0.025);
    let world_pos = head.position + head.orientation * local_offset;
    // Rotate so marker's +y aligns with head's +z (mesh long axis points
    // forward from face).
    let marker_rot = head.orientation
        * UnitQuaternion::from_scaled_axis(NaVec3::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0));
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
    let thoracic = &phys.world.bodies[phys.thoracic_idx];
    let head = &phys.world.bodies[phys.head_idx];
    let q_rel = head.orientation * thoracic.orientation.inverse();
    let s = q_rel.scaled_axis();
    let flex_actual = s.dot(&NaVec3::new(1.0, 0.0, 0.0));
    let bend_actual = s.dot(&NaVec3::new(0.0, 0.0, -1.0));
    let rot_actual  = s.dot(&NaVec3::new(0.0, 1.0, 0.0));
    let (flex_t, bend_t, rot_t, phase) = cerv_targets(phys.sim_time);
    for mut text in &mut query {
        text.sections[0].value = format!(
            "{}\nt: {:.2} s  (cycle {:.0} s)\n\n  axis              target → actual\n  flex/ext  : {:+.2}  →  {:+.2}\n  bend (R+) : {:+.2}  →  {:+.2}\n  rot  (R+) : {:+.2}  →  {:+.2}",
            phase, phys.sim_time, SCHEDULE_PERIOD,
            flex_t, flex_actual,
            bend_t, bend_actual,
            rot_t, rot_actual,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Cervical demo (full ROM, 3-DOF neck)".into(),
                resolution: (1024.0, 768.0).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(Time::<Fixed>::from_hz(PHYSICS_HZ))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, physics_step)
        .add_systems(Update, (sync_transforms, sync_nose_marker, update_readout))
        .run();
}
