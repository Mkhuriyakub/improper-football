// Visual demo of Phase 1 anatomical ankle — static-balance FAILURE mode.
//
// This binary builds the same multi-body ankle assembly (talocrural +
// subtalar + MTP) as `ankle_anatomical.rs` and runs the static-balance
// scenario (PD targets at 0, gravity on, foot on ground).
//
// The chain starts upright, holds for ~1.4 s, then tips forward as the
// single-capsule foot rolls. This is a known, structural Phase 1
// limitation:
//   - Single-capsule foot rolls (one end lifts off ground)
//   - PDs hold joint angle at neutral, not absolute chain tilt
//
// Real 2D footprint requires a multi-capsule foot or new contact
// primitive; absolute-tilt control requires a whole-body balance
// controller. Both are deferred to later phases.
//
// Constants, helpers, and build_world are duplicated from
// `ankle_anatomical.rs` (matching the per-binary self-contained
// pattern used throughout `src/bin/*.rs`). Open follow-up: extract
// these to the `physics` crate.

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

// ─── Simulation timing ────────────────────────────────────────────────

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

// ─── Body parameters (from spec §3) ───────────────────────────────────

const SHIN_MASS:    f64 = 3.00;
const SHIN_HALF_H:  f64 = 0.21;
const SHIN_RADIUS:  f64 = 0.05;

const TALUS_MASS:   f64 = 0.05;
const TALUS_HALF_H: f64 = 0.025;
const TALUS_RADIUS: f64 = 0.020;

const FOOT_MASS:    f64 = 0.85;
const FOOT_HALF_H:  f64 = 0.090;
const FOOT_RADIUS:  f64 = 0.032;

const FOREFOOT_MASS:    f64 = 0.10;
const FOREFOOT_HALF_H:  f64 = 0.035;
const FOREFOOT_RADIUS:  f64 = 0.020;

const STICK_MASS:    f64 = 7.00;
const STICK_HALF_H:  f64 = 0.40;
const STICK_RADIUS:  f64 = 0.08;

// ─── Joint parameters (from spec §4) ──────────────────────────────────

const TALOCRURAL_JOINT_K: f64 = 60_000.0;
const TALOCRURAL_JOINT_C: f64 =    150.0;
const TALOCRURAL_PRIMARY_KP: f64 = 80.0;
const TALOCRURAL_PRIMARY_KD: f64 =  1.0;
const TALOCRURAL_PASSIVE_K:  f64 =  4.0;
const TALOCRURAL_PASSIVE_C:  f64 =  1.2;
const TALOCRURAL_ROM_MIN: f64 = -0.35;
const TALOCRURAL_ROM_MAX: f64 = 0.70;
const TALOCRURAL_LIMIT_K: f64 = 10_000.0;
const TALOCRURAL_LIMIT_EXP: f64 = 2.0;
const TALOCRURAL_OFF_AXIS_K:    f64 = 200.0;
const TALOCRURAL_OFF_AXIS_LIM:  f64 = 0.05;
const TALOCRURAL_OFF_AXIS_LIMIT_K: f64 = 10_000.0;

const SUBTALAR_JOINT_K: f64 = 60_000.0;
const SUBTALAR_JOINT_C: f64 =    150.0;
const SUBTALAR_PRIMARY_KP: f64 = 50.0;
const SUBTALAR_PRIMARY_KD: f64 =  2.0;
const SUBTALAR_PASSIVE_K:  f64 =  3.5;
const SUBTALAR_PASSIVE_C:  f64 =  1.0;
const SUBTALAR_ROM_MIN: f64 = -0.26;
const SUBTALAR_ROM_MAX: f64 = 0.52;
const SUBTALAR_LIMIT_K: f64 = 8_000.0;
const SUBTALAR_LIMIT_EXP: f64 = 2.0;
const SUBTALAR_TWIST_K: f64 = 8.0;
const SUBTALAR_TWIST_C: f64 = 0.5;
const SUBTALAR_TWIST_LIM: f64 = 0.10;
const SUBTALAR_TWIST_LIMIT_K: f64 = 10_000.0;
const SUBTALAR_OFF_AXIS_K:    f64 = 200.0;
const SUBTALAR_OFF_AXIS_LIM:  f64 = 0.05;
const SUBTALAR_OFF_AXIS_LIMIT_K: f64 = 10_000.0;

const MTP_JOINT_K: f64 = 60_000.0;
const MTP_JOINT_C: f64 =    150.0;
const MTP_PRIMARY_KP: f64 = 30.0;
const MTP_PRIMARY_KD: f64 =  0.5;
const MTP_PASSIVE_K:  f64 =  5.0;
const MTP_PASSIVE_C:  f64 =  1.5;
const MTP_ROM_MIN: f64 = -0.35;
const MTP_ROM_MAX: f64 = 0.87;
const MTP_LIMIT_K: f64 = 5_000.0;
const MTP_LIMIT_EXP: f64 = 2.0;
const MTP_OFF_AXIS_K:    f64 = 100.0;
const MTP_OFF_AXIS_LIM:  f64 = 0.05;
const MTP_OFF_AXIS_LIMIT_K: f64 = 5_000.0;

const WELD_POS_K: f64 = 1_000_000.0;
const WELD_POS_C: f64 =     2_000.0;
const WELD_ANG_K: f64 =    50_000.0;
const WELD_ANG_C: f64 =       500.0;

// ─── Helpers ──────────────────────────────────────────────────────────

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

fn weld_bodies(
    world: &mut XpbdWorld, body_a: usize, body_b: usize,
    anchor_a: NaVec3, anchor_b: NaVec3,
) {
    add_elastic_joint(world, body_a, body_b, anchor_a, anchor_b, WELD_POS_K, WELD_POS_C);
    for axis in [NaVec3::x(), NaVec3::y(), NaVec3::z()] {
        world.angular_springs.push(
            AngularSpring::new(body_a, body_b, axis, WELD_ANG_K, 0.0));
        world.angular_dampers.push(
            AngularDamper::new(body_a, body_b, axis, WELD_ANG_C));
    }
}

fn joint_angle(
    world: &XpbdWorld, body_a: usize, body_b: usize, axis_local_a: NaVec3,
) -> f64 {
    let q_b_in_a = world.bodies[body_a].orientation.inverse()
                 * world.bodies[body_b].orientation;
    swing_twist_angle(q_b_in_a, axis_local_a)
}

#[derive(Clone, Copy)]
struct AnkleHandles {
    shin_idx: usize,
    talus_idx: usize,
    foot_idx: usize,
    forefoot_idx: usize,
    stick_idx: usize,
    #[allow(dead_code)]
    talocrural_pd_idx: usize,
    #[allow(dead_code)]
    subtalar_pd_idx: usize,
    #[allow(dead_code)]
    mtp_pd_idx: usize,
}

fn build_world() -> (XpbdWorld, AnkleHandles) {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let id = UnitQuaternion::identity();

    let foot_baseline = UnitQuaternion::from_scaled_axis(
        NaVec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    let mut foot = RigidBody::new_capsule(FOOT_MASS, FOOT_HALF_H, FOOT_RADIUS);
    foot.position = NaVec3::new(0.0, FOOT_RADIUS, 0.09);
    foot.orientation = foot_baseline;
    let foot_idx = world.add_body(foot);

    let mut forefoot = RigidBody::new_capsule(FOREFOOT_MASS, FOREFOOT_HALF_H, FOREFOOT_RADIUS);
    let (pos, orient) = place_child(&world.bodies[foot_idx],
        NaVec3::new(0.0, -FOOT_HALF_H, 0.0),
        NaVec3::new(0.0, FOREFOOT_HALF_H, 0.0),
        id);
    forefoot.position = pos;
    forefoot.orientation = orient;
    let forefoot_idx = world.add_body(forefoot);

    let mut talus = RigidBody::new_capsule(TALUS_MASS, TALUS_HALF_H, TALUS_RADIUS);
    talus.position = NaVec3::new(0.0, FOOT_RADIUS + TALUS_HALF_H, 0.0);
    talus.orientation = id;
    let talus_idx = world.add_body(talus);

    let mut shin = RigidBody::new_capsule(SHIN_MASS, SHIN_HALF_H, SHIN_RADIUS);
    let talocrural_world_y = FOOT_RADIUS + 2.0 * TALUS_HALF_H;
    shin.position = NaVec3::new(0.0, talocrural_world_y + SHIN_HALF_H, 0.0);
    shin.orientation = id;
    let shin_idx = world.add_body(shin);

    // ── Talocrural joint (shin ↔ talus) ─────────────────────────────
    add_elastic_joint(&mut world, shin_idx, talus_idx,
        NaVec3::new(0.0, -SHIN_HALF_H, 0.0),
        NaVec3::new(0.0, TALUS_HALF_H, 0.0),
        TALOCRURAL_JOINT_K, TALOCRURAL_JOINT_C);

    let tc_primary = NaVec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        shin_idx, talus_idx, tc_primary,
        TALOCRURAL_ROM_MIN, TALOCRURAL_ROM_MAX,
        TALOCRURAL_LIMIT_K, TALOCRURAL_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(
        shin_idx, talus_idx, tc_primary, TALOCRURAL_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(
        shin_idx, talus_idx, tc_primary, TALOCRURAL_PASSIVE_K, 0.0));
    let talocrural_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        shin_idx, talus_idx, tc_primary,
        TALOCRURAL_PRIMARY_KP, TALOCRURAL_PRIMARY_KD));

    for off_axis in [NaVec3::y(), NaVec3::z()] {
        world.angular_soft_limits.push(AngularSoftLimit::new(
            shin_idx, talus_idx, off_axis,
            -TALOCRURAL_OFF_AXIS_LIM, TALOCRURAL_OFF_AXIS_LIM,
            TALOCRURAL_OFF_AXIS_LIMIT_K, TALOCRURAL_LIMIT_EXP));
        world.angular_dampers.push(AngularDamper::new(
            shin_idx, talus_idx, off_axis, TALOCRURAL_PASSIVE_C));
        world.angular_springs.push(AngularSpring::new(
            shin_idx, talus_idx, off_axis, TALOCRURAL_OFF_AXIS_K, 0.0));
    }

    // ── Subtalar joint (talus ↔ foot) ───────────────────────────────
    add_elastic_joint(&mut world, talus_idx, foot_idx,
        NaVec3::new(0.0, -TALUS_HALF_H, 0.0),
        NaVec3::new(0.0, FOOT_HALF_H, 0.0),
        SUBTALAR_JOINT_K, SUBTALAR_JOINT_C);

    let st_primary = NaVec3::new(0.0, 0.0, 1.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        talus_idx, foot_idx, st_primary,
        SUBTALAR_ROM_MIN, SUBTALAR_ROM_MAX,
        SUBTALAR_LIMIT_K, SUBTALAR_LIMIT_EXP)
        .with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(
        talus_idx, foot_idx, st_primary, SUBTALAR_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(
        talus_idx, foot_idx, st_primary, SUBTALAR_PASSIVE_K, 0.0)
        .with_baseline(foot_baseline));
    let subtalar_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        talus_idx, foot_idx, st_primary,
        SUBTALAR_PRIMARY_KP, SUBTALAR_PRIMARY_KD)
        .with_baseline(foot_baseline));

    let st_twist = NaVec3::new(0.0, 1.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        talus_idx, foot_idx, st_twist,
        -SUBTALAR_TWIST_LIM, SUBTALAR_TWIST_LIM,
        SUBTALAR_TWIST_LIMIT_K, SUBTALAR_LIMIT_EXP)
        .with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(
        talus_idx, foot_idx, st_twist, SUBTALAR_TWIST_C));
    world.angular_springs.push(AngularSpring::new(
        talus_idx, foot_idx, st_twist, SUBTALAR_TWIST_K, 0.0)
        .with_baseline(foot_baseline));

    let st_locked = NaVec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        talus_idx, foot_idx, st_locked,
        -SUBTALAR_OFF_AXIS_LIM, SUBTALAR_OFF_AXIS_LIM,
        SUBTALAR_OFF_AXIS_LIMIT_K, SUBTALAR_LIMIT_EXP)
        .with_baseline(foot_baseline));
    world.angular_dampers.push(AngularDamper::new(
        talus_idx, foot_idx, st_locked, SUBTALAR_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(
        talus_idx, foot_idx, st_locked, SUBTALAR_OFF_AXIS_K, 0.0)
        .with_baseline(foot_baseline));

    // ── MTP joint (foot ↔ forefoot) ─────────────────────────────────
    add_elastic_joint(&mut world, foot_idx, forefoot_idx,
        NaVec3::new(0.0, -FOOT_HALF_H, 0.0),
        NaVec3::new(0.0, FOREFOOT_HALF_H, 0.0),
        MTP_JOINT_K, MTP_JOINT_C);

    let mtp_primary = NaVec3::new(1.0, 0.0, 0.0);
    world.angular_soft_limits.push(AngularSoftLimit::new(
        foot_idx, forefoot_idx, mtp_primary,
        MTP_ROM_MIN, MTP_ROM_MAX,
        MTP_LIMIT_K, MTP_LIMIT_EXP));
    world.angular_dampers.push(AngularDamper::new(
        foot_idx, forefoot_idx, mtp_primary, MTP_PASSIVE_C));
    world.angular_springs.push(AngularSpring::new(
        foot_idx, forefoot_idx, mtp_primary, MTP_PASSIVE_K, 0.0));
    let mtp_pd_idx = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(AngularPDController::new(
        foot_idx, forefoot_idx, mtp_primary,
        MTP_PRIMARY_KP, MTP_PRIMARY_KD));

    for off_axis in [NaVec3::y(), NaVec3::z()] {
        world.angular_soft_limits.push(AngularSoftLimit::new(
            foot_idx, forefoot_idx, off_axis,
            -MTP_OFF_AXIS_LIM, MTP_OFF_AXIS_LIM,
            MTP_OFF_AXIS_LIMIT_K, MTP_LIMIT_EXP));
        world.angular_dampers.push(AngularDamper::new(
            foot_idx, forefoot_idx, off_axis, MTP_PASSIVE_C));
        world.angular_springs.push(AngularSpring::new(
            foot_idx, forefoot_idx, off_axis, MTP_OFF_AXIS_K, 0.0));
    }

    let mut stick = RigidBody::new_capsule(STICK_MASS, STICK_HALF_H, STICK_RADIUS);
    let shin_top_y = talocrural_world_y + 2.0 * SHIN_HALF_H;
    stick.position = NaVec3::new(0.0, shin_top_y + STICK_HALF_H, 0.0);
    stick.orientation = id;
    let stick_idx = world.add_body(stick);

    weld_bodies(&mut world, shin_idx, stick_idx,
        NaVec3::new(0.0, SHIN_HALF_H, 0.0),
        NaVec3::new(0.0, -STICK_HALF_H, 0.0));

    (world, AnkleHandles {
        shin_idx, talus_idx, foot_idx, forefoot_idx, stick_idx,
        talocrural_pd_idx, subtalar_pd_idx, mtp_pd_idx,
    })
}

// ─── Bevy plumbing ────────────────────────────────────────────────────

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    sim_time: f64,
    handles: AnkleHandles,
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
    let (world, handles) = build_world();

    // Materials.
    let foot_mat     = materials.add(StandardMaterial {
        base_color: Color::srgb(0.55, 0.35, 0.25), ..default() });
    let forefoot_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.65, 0.45, 0.30), ..default() });
    let talus_mat    = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.78, 0.55), ..default() });
    let shin_mat     = materials.add(StandardMaterial {
        base_color: Color::srgb(0.80, 0.62, 0.48), ..default() });
    let stick_mat    = materials.add(StandardMaterial {
        base_color: Color::srgb(0.45, 0.50, 0.62), ..default() });

    // Meshes (capsule total length = 2 * half_h).
    let foot_mesh     = meshes.add(Capsule3d::new(
        FOOT_RADIUS as f32, (2.0 * FOOT_HALF_H) as f32));
    let forefoot_mesh = meshes.add(Capsule3d::new(
        FOREFOOT_RADIUS as f32, (2.0 * FOREFOOT_HALF_H) as f32));
    let talus_mesh    = meshes.add(Capsule3d::new(
        TALUS_RADIUS as f32, (2.0 * TALUS_HALF_H) as f32));
    let shin_mesh     = meshes.add(Capsule3d::new(
        SHIN_RADIUS as f32, (2.0 * SHIN_HALF_H) as f32));
    let stick_mesh    = meshes.add(Capsule3d::new(
        STICK_RADIUS as f32, (2.0 * STICK_HALF_H) as f32));

    commands.spawn((PbrBundle {
        mesh: foot_mesh, material: foot_mat,
        transform: rigid_body_to_transform(&world.bodies[handles.foot_idx]),
        ..default()
    }, BodyRef(handles.foot_idx)));
    commands.spawn((PbrBundle {
        mesh: forefoot_mesh, material: forefoot_mat,
        transform: rigid_body_to_transform(&world.bodies[handles.forefoot_idx]),
        ..default()
    }, BodyRef(handles.forefoot_idx)));
    commands.spawn((PbrBundle {
        mesh: talus_mesh, material: talus_mat,
        transform: rigid_body_to_transform(&world.bodies[handles.talus_idx]),
        ..default()
    }, BodyRef(handles.talus_idx)));
    commands.spawn((PbrBundle {
        mesh: shin_mesh, material: shin_mat,
        transform: rigid_body_to_transform(&world.bodies[handles.shin_idx]),
        ..default()
    }, BodyRef(handles.shin_idx)));
    commands.spawn((PbrBundle {
        mesh: stick_mesh, material: stick_mat,
        transform: rigid_body_to_transform(&world.bodies[handles.stick_idx]),
        ..default()
    }, BodyRef(handles.stick_idx)));

    // Ground.
    let ground_mesh = meshes.add(Plane3d::default().mesh().size(4.0, 4.0));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground_mesh, material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0), ..default()
    });

    // Side camera: look along +x at the assembly so the user sees the
    // forward (+z) tip. The chain stands at roughly y ∈ [0, 1.3] m,
    // forward foot tip near z ≈ 0.2.
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(2.6, 0.95, 0.6)
            .looking_at(Vec3::new(0.0, 0.7, 0.1), Vec3::Y),
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

    commands.insert_resource(PhysicsWorld { world, sim_time: 0.0, handles });
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    for _ in 0..SOLVER_ITERATIONS {
        phys.world.substep(SUB_DT);
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
    let h = phys.handles;
    let talocrural = joint_angle(&phys.world, h.shin_idx, h.talus_idx,
        NaVec3::new(1.0, 0.0, 0.0)).to_degrees();
    let subtalar = joint_angle(&phys.world, h.talus_idx, h.foot_idx,
        NaVec3::new(0.0, 0.0, 1.0)).to_degrees();
    let mtp = joint_angle(&phys.world, h.foot_idx, h.forefoot_idx,
        NaVec3::new(1.0, 0.0, 0.0)).to_degrees();
    let stick_tilt = phys.world.bodies[h.stick_idx]
        .orientation.scaled_axis().norm().to_degrees();

    for mut text in &mut q {
        text.sections[0].value = format!(
            "Phase 1 anatomical ankle — static balance FAILURE demo\n\
             \n\
             t: {:.2} s\n\
             \n\
             talocrural (PF/DF):  {:+7.2}°\n\
             subtalar  (INV/EVR):  {:+7.2}°\n\
             MTP       (toe flex): {:+7.2}°\n\
             \n\
             stick tilt:           {:+7.2}°   (PASS threshold: 5°)\n\
             \n\
             PD targets at 0. Chain starts upright and holds for ~1.4 s,\n\
             then tips forward as the single-capsule foot rolls.\n\
             \n\
             Known Phase 1 limitation:\n\
              - Single-capsule foot has 1D ground contact (rolls).\n\
              - PDs hold joint angle at neutral, not absolute tilt.\n\
             Both deferred to later phases.",
            phys.sim_time,
            talocrural, subtalar, mtp, stick_tilt,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Phase 1 anatomical ankle — static balance FAILURE demo".into(),
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
