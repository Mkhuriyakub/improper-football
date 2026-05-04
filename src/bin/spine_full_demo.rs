// Full spine assembly: pelvis (pinned) → lumbar (3-DOF) → thoracic
// → cervical (3-DOF) → head. 4 bodies, 2 joint regions.
//
// Sequential ROM-only schedule. Each region sweeps through its anatomical
// range independently — no coordinated multi-joint poses (those come
// later, in concert with limbs). The point is to verify the chain is
// wired correctly: lumbar motion carries the head along, cervical motion
// is relative to the (possibly moving) thoracic, and both joints settle
// to upright when not driven.
//
// Schedule (52 s, looping):
//
//   LUMBAR phase:
//     0–6 s     lumbar flexion        0 → +0.90 → 0  (~51°)
//     6–10 s    lumbar extension      0 → -0.30 → 0
//     10–14 s   lumbar right bend     0 → +0.35 → 0
//     14–18 s   lumbar left bend      0 → -0.35 → 0
//     18–22 s   lumbar trunk circ.    flex/bend traced as a circle
//
//   CERVICAL phase:
//     22–26 s   cervical flexion      0 → +0.75 → 0  (chin to chest)
//     26–30 s   cervical extension    0 → -0.95 → 0  (look up)
//     30–34 s   cervical bend R/L     ±0.65 sweep
//     34–40 s   cervical rotation     ±1.20 sweep (look L/R)
//     40–46 s   head circumduction    flex/bend traced as a circle
//
//   COORDINATED:
//     46–52 s   look up + back arch   lumbar ext + cervical ext

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

// ─── Pelvis (pinned) ──────────────────────────────────────────────────
const PELVIS_HALF_H: f64 = 0.10;
const PELVIS_RADIUS: f64 = 0.18;
const PELVIS_REST_POS: NaVec3 = NaVec3::new(0.0, 1.00, 0.0);

// ─── Thoracic ─────────────────────────────────────────────────────────
const THORACIC_MASS:    f64 = 25.0;
const THORACIC_RADIUS:  f64 = 0.130;
const THORACIC_HALF_H:  f64 = 0.180;

// ─── Lumbar joint (pelvis ↔ thoracic) ─────────────────────────────────
const LUMBAR_JOINT_K: f64 = 300_000.0;
const LUMBAR_JOINT_C: f64 =     600.0;
const L_FLEX_K: f64 = 150.0;  const L_FLEX_C: f64 = 25.0;
const L_BEND_K: f64 = 150.0;  const L_BEND_C: f64 = 25.0;
const L_ROT_K:  f64 =  50.0;  const L_ROT_C:  f64 =  5.0;
const L_FLEX_MIN: f64 = -0.44;   const L_FLEX_MAX: f64 = 1.05;
const L_BEND_MIN: f64 = -0.44;   const L_BEND_MAX: f64 = 0.44;
const L_ROT_MIN:  f64 = -0.087;  const L_ROT_MAX:  f64 = 0.087;
const LUMBAR_LIMIT_K:   f64 = 200.0;
const LUMBAR_LIMIT_EXP: f64 =   2.0;

// Lumbar PD — KP bumped vs lumbar-only demo because the chain now carries
// the head as well. I_total at lumbar ≈ 2.06 kg·m² (thoracic + head with
// parallel-axis term to lumbar joint).
const LUMBAR_PD_KP: f64 = 1500.0;
const LUMBAR_PD_KD: f64 =  100.0;

// ─── Head ─────────────────────────────────────────────────────────────
const HEAD_MASS:    f64 = 5.0;
const HEAD_RADIUS:  f64 = 0.080;
const HEAD_HALF_H:  f64 = 0.050;

// ─── Cervical joint (thoracic ↔ head) ─────────────────────────────────
const CERV_JOINT_K: f64 = 80_000.0;
const CERV_JOINT_C: f64 =    200.0;
const C_FLEX_K: f64 = 5.0;  const C_FLEX_C: f64 = 0.60;
const C_BEND_K: f64 = 5.0;  const C_BEND_C: f64 = 0.60;
const C_ROT_K:  f64 = 3.0;  const C_ROT_C:  f64 = 0.50;
const C_FLEX_MIN: f64 = -1.05;   const C_FLEX_MAX: f64 = 0.87;
const C_BEND_MIN: f64 = -0.78;   const C_BEND_MAX: f64 = 0.78;
const C_ROT_MIN:  f64 = -1.40;   const C_ROT_MAX:  f64 = 1.40;
const CERV_LIMIT_K:   f64 = 80.0;
const CERV_LIMIT_EXP: f64 =  2.0;
const CERV_PD_KP: f64 = 60.0;
const CERV_PD_KD: f64 =  3.0;

const SCHEDULE_PERIOD: f64 = 52.0;

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
    world: &mut XpbdWorld, parent: usize, child: usize,
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
    world: &mut XpbdWorld, parent: usize, child: usize,
    axis_local_a: NaVec3, min_angle: f64, max_angle: f64,
    stiffness: f64, damping: f64, neutral: f64,
    limit_stiffness: f64, limit_exp: f64,
) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, axis_local_a, min_angle, max_angle, limit_stiffness, limit_exp));
    world.angular_dampers.push(AngularDamper::new(parent, child, axis_local_a, damping));
    world.angular_springs.push(AngularSpring::new(parent, child, axis_local_a, stiffness, neutral));
}

fn half_cosine(t: f64, duration: f64, peak: f64) -> f64 {
    let phase = (t / duration).clamp(0.0, 1.0);
    peak * 0.5 * (1.0 - (std::f64::consts::TAU * phase).cos())
}

#[derive(Clone, Copy, Default)]
struct SpineTargets {
    l_flex: f64, l_bend: f64, l_rot: f64,
    c_flex: f64, c_bend: f64, c_rot: f64,
}

fn spine_targets(t: f64) -> (SpineTargets, &'static str) {
    let t = t.rem_euclid(SCHEDULE_PERIOD);
    let zero = SpineTargets::default();

    if t < 6.0 {
        (SpineTargets { l_flex: half_cosine(t, 6.0, 0.90), ..zero }, "lumbar flexion")
    } else if t < 10.0 {
        (SpineTargets { l_flex: half_cosine(t - 6.0, 4.0, -0.30), ..zero }, "lumbar extension")
    } else if t < 14.0 {
        (SpineTargets { l_bend: half_cosine(t - 10.0, 4.0, 0.35), ..zero }, "lumbar right bend")
    } else if t < 18.0 {
        (SpineTargets { l_bend: half_cosine(t - 14.0, 4.0, -0.35), ..zero }, "lumbar left bend")
    } else if t < 22.0 {
        let local = (t - 18.0).clamp(0.0, 4.0);
        let env = (std::f64::consts::PI * local / 4.0).sin();
        let phase = std::f64::consts::TAU * local / 4.0;
        (SpineTargets {
            l_flex: 0.30 * env + 0.30 * env * phase.sin(),
            l_bend: 0.30 * env * phase.cos(),
            ..zero
        }, "lumbar trunk circumduction")
    } else if t < 26.0 {
        (SpineTargets { c_flex: half_cosine(t - 22.0, 4.0, 0.75), ..zero }, "cervical flexion (chin-to-chest)")
    } else if t < 30.0 {
        (SpineTargets { c_flex: half_cosine(t - 26.0, 4.0, -0.95), ..zero }, "cervical extension (look up)")
    } else if t < 34.0 {
        // Right then left bend, each as its own 2 s half-cosine bump
        // (0 → ±0.65 → 0). Both branches end at 0 with zero velocity, so
        // the right→left transition has no jerk.
        let local = t - 30.0;
        let bend = if local < 2.0 {
            half_cosine(local, 2.0, 0.65)
        } else {
            half_cosine(local - 2.0, 2.0, -0.65)
        };
        (SpineTargets { c_bend: bend, ..zero }, "cervical lateral bend")
    } else if t < 40.0 {
        // Right then left rotation: each as its own 3 s half-cosine bump
        // (0 → ±1.20 → 0). Smooth at the right→left transition.
        let local = t - 34.0;
        let rot = if local < 3.0 {
            half_cosine(local, 3.0, 1.20)
        } else {
            half_cosine(local - 3.0, 3.0, -1.20)
        };
        (SpineTargets { c_rot: rot, ..zero }, "cervical rotation (look L/R)")
    } else if t < 46.0 {
        // Head circumduction (chin tracing a circle).
        let local = (t - 40.0).clamp(0.0, 6.0);
        let env = (std::f64::consts::PI * local / 6.0).sin();
        let phase = std::f64::consts::TAU * local / 6.0;
        (SpineTargets {
            c_flex: 0.55 * env * phase.sin(),
            c_bend: 0.55 * env * phase.cos(),
            ..zero
        }, "head circumduction")
    } else {
        // Coordinated: lumbar extension + cervical extension (back arch
        // with head tilted up — looking at the ceiling).
        let local = t - 46.0;
        (SpineTargets {
            l_flex: half_cosine(local, 6.0, -0.25),
            c_flex: half_cosine(local, 6.0, -0.85),
            ..zero
        }, "back arch + look up")
    }
}

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    pelvis_idx: usize,
    thoracic_idx: usize,
    head_idx: usize,
    sim_time: f64,
    pd_l_flex: usize, pd_l_bend: usize, pd_l_rot: usize,
    pd_c_flex: usize, pd_c_bend: usize, pd_c_rot: usize,
}

#[derive(Component)] struct BodyRef(usize);
#[derive(Component)] struct ShoulderMarker;
#[derive(Component)] struct NoseMarker;
#[derive(Component)] struct Readout;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();

    // Pelvis (pinned).
    let mut pelvis = RigidBody::new(0.0, capsule_inertia(15.0, PELVIS_RADIUS, PELVIS_HALF_H));
    pelvis.position = PELVIS_REST_POS;
    pelvis.orientation = UnitQuaternion::identity();
    let pelvis_idx = world.add_body(pelvis);

    // Thoracic — sits on top of pelvis.
    let lumbar_anchor_pelvis = NaVec3::new(0.0, PELVIS_HALF_H, 0.0);
    let mut thoracic = RigidBody::new(THORACIC_MASS,
        capsule_inertia(THORACIC_MASS, THORACIC_RADIUS, THORACIC_HALF_H));
    let (pos, orient) = place_child(&world.bodies[pelvis_idx],
        lumbar_anchor_pelvis,
        NaVec3::new(0.0, -THORACIC_HALF_H, 0.0),
        UnitQuaternion::identity());
    thoracic.position = pos; thoracic.orientation = orient;
    let thoracic_idx = world.add_body(thoracic);

    add_elastic_joint(&mut world, pelvis_idx, thoracic_idx,
        lumbar_anchor_pelvis, NaVec3::new(0.0, -THORACIC_HALF_H, 0.0),
        LUMBAR_JOINT_K, LUMBAR_JOINT_C);

    // Lumbar axes — same convention as lumbar_demo (anatomical mapping).
    let l_flex_axis = NaVec3::new(1.0, 0.0, 0.0);
    let l_bend_axis = NaVec3::new(0.0, 0.0, -1.0);
    let l_rot_axis  = NaVec3::new(0.0, 1.0, 0.0);
    add_axis(&mut world, pelvis_idx, thoracic_idx, l_flex_axis,
        L_FLEX_MIN, L_FLEX_MAX, L_FLEX_K, L_FLEX_C, 0.0,
        LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP);
    add_axis(&mut world, pelvis_idx, thoracic_idx, l_bend_axis,
        L_BEND_MIN, L_BEND_MAX, L_BEND_K, L_BEND_C, 0.0,
        LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP);
    add_axis(&mut world, pelvis_idx, thoracic_idx, l_rot_axis,
        L_ROT_MIN, L_ROT_MAX, L_ROT_K, L_ROT_C, 0.0,
        LUMBAR_LIMIT_K, LUMBAR_LIMIT_EXP);

    let pd_l_flex = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thoracic_idx, l_flex_axis,
            LUMBAR_PD_KP, LUMBAR_PD_KD).with_target(0.0));
    let pd_l_bend = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thoracic_idx, l_bend_axis,
            LUMBAR_PD_KP, LUMBAR_PD_KD).with_target(0.0));
    let pd_l_rot = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(pelvis_idx, thoracic_idx, l_rot_axis,
            LUMBAR_PD_KP, LUMBAR_PD_KD).with_target(0.0));

    // Head — sits on top of thoracic.
    let cerv_anchor_thoracic = NaVec3::new(0.0, THORACIC_HALF_H, 0.0);
    let mut head = RigidBody::new(HEAD_MASS,
        capsule_inertia(HEAD_MASS, HEAD_RADIUS, HEAD_HALF_H));
    let (pos, orient) = place_child(&world.bodies[thoracic_idx],
        cerv_anchor_thoracic,
        NaVec3::new(0.0, -HEAD_HALF_H, 0.0),
        UnitQuaternion::identity());
    head.position = pos; head.orientation = orient;
    let head_idx = world.add_body(head);

    add_elastic_joint(&mut world, thoracic_idx, head_idx,
        cerv_anchor_thoracic, NaVec3::new(0.0, -HEAD_HALF_H, 0.0),
        CERV_JOINT_K, CERV_JOINT_C);

    let c_flex_axis = NaVec3::new(1.0, 0.0, 0.0);
    let c_bend_axis = NaVec3::new(0.0, 0.0, -1.0);
    let c_rot_axis  = NaVec3::new(0.0, 1.0, 0.0);
    add_axis(&mut world, thoracic_idx, head_idx, c_flex_axis,
        C_FLEX_MIN, C_FLEX_MAX, C_FLEX_K, C_FLEX_C, 0.0,
        CERV_LIMIT_K, CERV_LIMIT_EXP);
    add_axis(&mut world, thoracic_idx, head_idx, c_bend_axis,
        C_BEND_MIN, C_BEND_MAX, C_BEND_K, C_BEND_C, 0.0,
        CERV_LIMIT_K, CERV_LIMIT_EXP);
    add_axis(&mut world, thoracic_idx, head_idx, c_rot_axis,
        C_ROT_MIN, C_ROT_MAX, C_ROT_K, C_ROT_C, 0.0,
        CERV_LIMIT_K, CERV_LIMIT_EXP);

    let pd_c_flex = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thoracic_idx, head_idx, c_flex_axis,
            CERV_PD_KP, CERV_PD_KD).with_target(0.0));
    let pd_c_bend = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thoracic_idx, head_idx, c_bend_axis,
            CERV_PD_KP, CERV_PD_KD).with_target(0.0));
    let pd_c_rot = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(thoracic_idx, head_idx, c_rot_axis,
            CERV_PD_KP, CERV_PD_KD).with_target(0.0));

    // Meshes & materials.
    let pelvis_mesh   = meshes.add(Capsule3d::new(PELVIS_RADIUS as f32, (2.0 * PELVIS_HALF_H) as f32));
    let thoracic_mesh = meshes.add(Capsule3d::new(THORACIC_RADIUS as f32, (2.0 * THORACIC_HALF_H) as f32));
    let head_mesh     = meshes.add(Capsule3d::new(HEAD_RADIUS as f32, (2.0 * HEAD_HALF_H) as f32));

    let pelvis_mat   = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.55, 0.62), ..default() });
    let thoracic_mat = materials.add(StandardMaterial { base_color: Color::srgb(0.78, 0.62, 0.50), ..default() });
    let head_mat     = materials.add(StandardMaterial { base_color: Color::srgb(0.92, 0.78, 0.65), ..default() });

    commands.spawn((PbrBundle {
        mesh: pelvis_mesh, material: pelvis_mat,
        transform: rigid_body_to_transform(&world.bodies[pelvis_idx]), ..default()
    }, BodyRef(pelvis_idx)));
    commands.spawn((PbrBundle {
        mesh: thoracic_mesh, material: thoracic_mat,
        transform: rigid_body_to_transform(&world.bodies[thoracic_idx]), ..default()
    }, BodyRef(thoracic_idx)));
    commands.spawn((PbrBundle {
        mesh: head_mesh, material: head_mat,
        transform: rigid_body_to_transform(&world.bodies[head_idx]), ..default()
    }, BodyRef(head_idx)));

    // Shoulder marker on top of thoracic.
    let shoulder_mesh = meshes.add(Capsule3d::new(0.018, 0.18));
    let shoulder_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.30, 0.40, 0.65), ..default() });
    commands.spawn((PbrBundle {
        mesh: shoulder_mesh, material: shoulder_mat,
        transform: Transform::default(), ..default()
    }, ShoulderMarker));

    // Nose marker on front of head.
    let nose_mesh = meshes.add(Capsule3d::new(0.012, 0.05));
    let nose_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.65, 0.30, 0.30), ..default() });
    commands.spawn((PbrBundle {
        mesh: nose_mesh, material: nose_mat,
        transform: Transform::default(), ..default()
    }, NoseMarker));

    let ground_mesh = meshes.add(Plane3d::default().mesh().size(3.0, 3.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0), ..default()
    });

    // Camera: framing pelvis-to-top-of-head.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(1.50, 1.30, 1.60)
            .looking_at(Vec3::new(0.0, 1.30, 0.0), Vec3::Y),
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
        TextStyle { font_size: 18.0, color: Color::WHITE, ..default() },
    )
    .with_style(Style {
        position_type: PositionType::Absolute,
        top: Val::Px(12.0), left: Val::Px(12.0), ..default()
    }), Readout));

    commands.insert_resource(PhysicsWorld {
        world, pelvis_idx, thoracic_idx, head_idx,
        sim_time: 0.0,
        pd_l_flex, pd_l_bend, pd_l_rot,
        pd_c_flex, pd_c_bend, pd_c_rot,
    });
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let (tg, _) = spine_targets(phys.sim_time);
    let pdlf = phys.pd_l_flex;
    let pdlb = phys.pd_l_bend;
    let pdlr = phys.pd_l_rot;
    let pdcf = phys.pd_c_flex;
    let pdcb = phys.pd_c_bend;
    let pdcr = phys.pd_c_rot;
    phys.world.angular_pd_controllers[pdlf].target_angle = tg.l_flex;
    phys.world.angular_pd_controllers[pdlb].target_angle = tg.l_bend;
    phys.world.angular_pd_controllers[pdlr].target_angle = tg.l_rot;
    phys.world.angular_pd_controllers[pdcf].target_angle = tg.c_flex;
    phys.world.angular_pd_controllers[pdcb].target_angle = tg.c_bend;
    phys.world.angular_pd_controllers[pdcr].target_angle = tg.c_rot;
    for _ in 0..SOLVER_ITERATIONS {
        phys.world.substep(SUB_DT);
    }
    phys.sim_time += DT;
}

fn sync_transforms(
    phys: Res<PhysicsWorld>,
    mut q_body: Query<(&BodyRef, &mut Transform), (Without<ShoulderMarker>, Without<NoseMarker>)>,
    mut q_shoulder: Query<&mut Transform, (With<ShoulderMarker>, Without<BodyRef>, Without<NoseMarker>)>,
    mut q_nose: Query<&mut Transform, (With<NoseMarker>, Without<BodyRef>, Without<ShoulderMarker>)>,
) {
    for (body_ref, mut transform) in &mut q_body {
        *transform = rigid_body_to_transform(&phys.world.bodies[body_ref.0]);
    }

    let thoracic = &phys.world.bodies[phys.thoracic_idx];
    let local_offset = NaVec3::new(0.0, THORACIC_HALF_H, 0.0);
    let world_pos = thoracic.position + thoracic.orientation * local_offset;
    let marker_rot = thoracic.orientation
        * UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, -std::f64::consts::FRAC_PI_2));
    for mut transform in &mut q_shoulder {
        *transform = Transform {
            translation: nalgebra_vec3_to_bevy(world_pos),
            rotation: nalgebra_quat_to_bevy(marker_rot),
            scale: Vec3::ONE,
        };
    }

    let head = &phys.world.bodies[phys.head_idx];
    let local_offset = NaVec3::new(0.0, 0.0, HEAD_RADIUS + 0.025);
    let world_pos = head.position + head.orientation * local_offset;
    let marker_rot = head.orientation
        * UnitQuaternion::from_scaled_axis(NaVec3::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0));
    for mut transform in &mut q_nose {
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
    let thoracic = &phys.world.bodies[phys.thoracic_idx];
    let head = &phys.world.bodies[phys.head_idx];
    let l_rel = (thoracic.orientation * pelvis.orientation.inverse()).scaled_axis();
    let c_rel = (head.orientation * thoracic.orientation.inverse()).scaled_axis();
    let l_flex = l_rel.dot(&NaVec3::new(1.0, 0.0, 0.0));
    let l_bend = l_rel.dot(&NaVec3::new(0.0, 0.0, -1.0));
    let l_rot  = l_rel.dot(&NaVec3::new(0.0, 1.0, 0.0));
    let c_flex = c_rel.dot(&NaVec3::new(1.0, 0.0, 0.0));
    let c_bend = c_rel.dot(&NaVec3::new(0.0, 0.0, -1.0));
    let c_rot  = c_rel.dot(&NaVec3::new(0.0, 1.0, 0.0));
    let (tg, phase) = spine_targets(phys.sim_time);
    for mut text in &mut query {
        text.sections[0].value = format!(
            "{}\nt: {:.1} s  (cycle {:.0} s)\n\n  axis              target → actual\n  lumbar flex : {:+.2}  →  {:+.2}\n  lumbar bend : {:+.2}  →  {:+.2}\n  lumbar rot  : {:+.3}  →  {:+.3}\n  cerv. flex  : {:+.2}  →  {:+.2}\n  cerv. bend  : {:+.2}  →  {:+.2}\n  cerv. rot   : {:+.2}  →  {:+.2}",
            phase, phys.sim_time, SCHEDULE_PERIOD,
            tg.l_flex, l_flex,
            tg.l_bend, l_bend,
            tg.l_rot, l_rot,
            tg.c_flex, c_flex,
            tg.c_bend, c_bend,
            tg.c_rot, c_rot,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Spine assembly (pelvis → lumbar → thoracic → cervical → head)".into(),
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
