// Visual demo of the three-phalanx finger with full passive biomechanics.
//
// Mirrors the validated `finger_compliant.rs` setup: hand + proximal/middle/
// distal phalanges connected by elastic joints (PositionalSpring + damper),
// each joint with its angular biomech (soft Z range limit, viscous damper,
// neutral-pose spring, gentle X/Y limits) and a flexor tendon coupling all
// three Z axes.
//
// Animation: every CYCLE_PERIOD seconds, the chain is reset to a deep curl
// (MCP=0.8, PIP=1.0, DIP=0.5). Between resets you can watch the passive
// biomechanics relax it toward its equilibrium pose (~0.26, 0.37, 0.30 rad).

use bevy::prelude::*;
use nalgebra::UnitQuaternion;

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::angular_damper::AngularDamper;
use physics::angular_spring::AngularSpring;
use physics::angular_soft_limit::AngularSoftLimit;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::tendon::{Tendon, TendonSpan};
use physics::math::{Vec3 as NaVec3, Mat3};
use improper_football::render_util::rigid_body_to_transform;

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

const HAND_REST_POS: NaVec3 = NaVec3::new(0.0, 1.0, 0.0);
const ANCHOR_KP_LIN: f64 = 50_000.0;
const ANCHOR_KP_ANG: f64 =    200.0;
const ANCHOR_KD_LIN: f64 =  1_000.0;
const ANCHOR_KD_ANG: f64 =      4.0;

const HAND_MASS:    f64 = 0.54;
const HAND_RADIUS:  f64 = 0.030;
const HAND_HALF_H:  f64 = 0.050;

const PROX_MASS:    f64 = 0.008;
const PROX_RADIUS:  f64 = 0.010;
const PROX_HALF_H:  f64 = 0.022;

const MID_MASS:     f64 = 0.005;
const MID_RADIUS:   f64 = 0.009;
const MID_HALF_H:   f64 = 0.012;

const DIST_MASS:    f64 = 0.003;
const DIST_RADIUS:  f64 = 0.008;
const DIST_HALF_H:  f64 = 0.008;

const JOINT_K: f64 = 1_000.0;
const C_HAND_PROX: f64 = 4.0;
const C_PROX_MID:  f64 = 2.5;
const C_MID_DIST:  f64 = 2.0;

struct JointBiomech {
    z_min: f64, z_max: f64,
    damping: f64, stiffness: f64, neutral: f64,
    limit_stiffness: f64,
}
const MCP: JointBiomech = JointBiomech {
    z_min: -0.52, z_max: 1.57,
    damping: 0.0004, stiffness: 0.05, neutral: 0.30,
    limit_stiffness: 0.5,
};
const PIP: JointBiomech = JointBiomech {
    z_min: -0.05, z_max: 2.09,
    damping: 0.00013, stiffness: 0.03, neutral: 0.40,
    limit_stiffness: 0.4,
};
const DIP: JointBiomech = JointBiomech {
    z_min: -0.05, z_max: 1.40,
    damping: 0.00006, stiffness: 0.02, neutral: 0.30,
    limit_stiffness: 0.3,
};

const TENDON_MOMENT_ARM: f64 = 0.005;
const TENDON_STIFFNESS:  f64 = 50.0;
const TENDON_DAMPING:    f64 = 2.0;
const TENDON_REST_EXCURSION: f64 = TENDON_MOMENT_ARM * (0.30 + 0.40 + 0.30);

const INITIAL_MCP: f64 = 0.8;
const INITIAL_PIP: f64 = 1.0;
const INITIAL_DIP: f64 = 0.5;

const CYCLE_PERIOD: f64 = 3.0;

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
    parent_idx: usize, child_idx: usize,
    parent_distal_local: NaVec3, child_proximal_local: NaVec3,
    damping: f64,
) {
    let mut spring = PositionalSpring::new(parent_idx, child_idx, JOINT_K);
    spring.local_anchor_a = parent_distal_local;
    spring.local_anchor_b = child_proximal_local;
    world.positional_springs.push(spring);

    let mut damp = PositionalDamper::new(parent_idx, child_idx, damping);
    damp.local_anchor_a = parent_distal_local;
    damp.local_anchor_b = child_proximal_local;
    world.positional_dampers.push(damp);
}

fn add_angular_biomech(world: &mut XpbdWorld, parent: usize, child: usize, b: &JointBiomech) {
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, NaVec3::z(),
        b.z_min, b.z_max, b.limit_stiffness, 2.0,
    ));
    world.angular_dampers.push(
        AngularDamper::new(parent, child, NaVec3::z(), b.damping));
    world.angular_springs.push(
        AngularSpring::new(parent, child, NaVec3::z(), b.stiffness, b.neutral));
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, NaVec3::x(), -0.10, 0.10, 0.3, 2.0,
    ));
    world.angular_soft_limits.push(AngularSoftLimit::new(
        parent, child, NaVec3::y(), -0.10, 0.10, 0.3, 2.0,
    ));
}

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    hand_idx: usize,
    prox_idx: usize,
    mid_idx: usize,
    dist_idx: usize,
    sim_time: f64,
    last_reset_time: f64,
}

#[derive(Component)]
struct BodyRef(usize);

#[derive(Component)]
struct AngleReadout;

fn reset_chain(phys: &mut PhysicsWorld) {
    let hand_orient = UnitQuaternion::identity();
    let h = &mut phys.world.bodies[phys.hand_idx];
    h.position = HAND_REST_POS;
    h.orientation = hand_orient;
    h.linear_velocity = NaVec3::zeros();
    h.angular_velocity = NaVec3::zeros();

    let mcp_rel = UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, INITIAL_MCP));
    let (pos, orient) = place_child(
        &phys.world.bodies[phys.hand_idx],
        NaVec3::new(0.0, -HAND_HALF_H, 0.0),
        NaVec3::new(0.0,  PROX_HALF_H, 0.0),
        mcp_rel,
    );
    let p = &mut phys.world.bodies[phys.prox_idx];
    p.position = pos;
    p.orientation = orient;
    p.linear_velocity = NaVec3::zeros();
    p.angular_velocity = NaVec3::zeros();

    let pip_rel = UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, INITIAL_PIP));
    let (pos, orient) = place_child(
        &phys.world.bodies[phys.prox_idx],
        NaVec3::new(0.0, -PROX_HALF_H, 0.0),
        NaVec3::new(0.0,  MID_HALF_H, 0.0),
        pip_rel,
    );
    let m = &mut phys.world.bodies[phys.mid_idx];
    m.position = pos;
    m.orientation = orient;
    m.linear_velocity = NaVec3::zeros();
    m.angular_velocity = NaVec3::zeros();

    let dip_rel = UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, INITIAL_DIP));
    let (pos, orient) = place_child(
        &phys.world.bodies[phys.mid_idx],
        NaVec3::new(0.0, -MID_HALF_H, 0.0),
        NaVec3::new(0.0,  DIST_HALF_H, 0.0),
        dip_rel,
    );
    let d = &mut phys.world.bodies[phys.dist_idx];
    d.position = pos;
    d.orientation = orient;
    d.linear_velocity = NaVec3::zeros();
    d.angular_velocity = NaVec3::zeros();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();

    let mut hand = RigidBody::new(HAND_MASS, capsule_inertia(HAND_MASS, HAND_RADIUS, HAND_HALF_H));
    hand.position = HAND_REST_POS;
    let hand_idx = world.add_body(hand);

    let mut prox = RigidBody::new(PROX_MASS, capsule_inertia(PROX_MASS, PROX_RADIUS, PROX_HALF_H));
    let mcp_rel = UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, INITIAL_MCP));
    let (pos, orient) = place_child(
        &world.bodies[hand_idx],
        NaVec3::new(0.0, -HAND_HALF_H, 0.0),
        NaVec3::new(0.0,  PROX_HALF_H, 0.0),
        mcp_rel,
    );
    prox.position = pos;
    prox.orientation = orient;
    let prox_idx = world.add_body(prox);

    let mut mid = RigidBody::new(MID_MASS, capsule_inertia(MID_MASS, MID_RADIUS, MID_HALF_H));
    let pip_rel = UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, INITIAL_PIP));
    let (pos, orient) = place_child(
        &world.bodies[prox_idx],
        NaVec3::new(0.0, -PROX_HALF_H, 0.0),
        NaVec3::new(0.0,  MID_HALF_H, 0.0),
        pip_rel,
    );
    mid.position = pos;
    mid.orientation = orient;
    let mid_idx = world.add_body(mid);

    let mut dist = RigidBody::new(DIST_MASS, capsule_inertia(DIST_MASS, DIST_RADIUS, DIST_HALF_H));
    let dip_rel = UnitQuaternion::from_scaled_axis(NaVec3::new(0.0, 0.0, INITIAL_DIP));
    let (pos, orient) = place_child(
        &world.bodies[mid_idx],
        NaVec3::new(0.0, -MID_HALF_H, 0.0),
        NaVec3::new(0.0,  DIST_HALF_H, 0.0),
        dip_rel,
    );
    dist.position = pos;
    dist.orientation = orient;
    let dist_idx = world.add_body(dist);

    add_elastic_joint(&mut world, hand_idx, prox_idx,
                      NaVec3::new(0.0, -HAND_HALF_H, 0.0),
                      NaVec3::new(0.0,  PROX_HALF_H, 0.0),
                      C_HAND_PROX);
    add_angular_biomech(&mut world, hand_idx, prox_idx, &MCP);

    add_elastic_joint(&mut world, prox_idx, mid_idx,
                      NaVec3::new(0.0, -PROX_HALF_H, 0.0),
                      NaVec3::new(0.0,  MID_HALF_H, 0.0),
                      C_PROX_MID);
    add_angular_biomech(&mut world, prox_idx, mid_idx, &PIP);

    add_elastic_joint(&mut world, mid_idx, dist_idx,
                      NaVec3::new(0.0, -MID_HALF_H, 0.0),
                      NaVec3::new(0.0,  DIST_HALF_H, 0.0),
                      C_MID_DIST);
    add_angular_biomech(&mut world, mid_idx, dist_idx, &DIP);

    world.tendons.push(Tendon::new(
        vec![
            TendonSpan { body_a: hand_idx, body_b: prox_idx, axis_local_a: NaVec3::z(), moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: prox_idx, body_b: mid_idx,  axis_local_a: NaVec3::z(), moment_arm: TENDON_MOMENT_ARM },
            TendonSpan { body_a: mid_idx,  body_b: dist_idx, axis_local_a: NaVec3::z(), moment_arm: TENDON_MOMENT_ARM },
        ],
        TENDON_REST_EXCURSION,
        TENDON_STIFFNESS,
        TENDON_DAMPING,
    ));

    let phys = PhysicsWorld {
        world,
        hand_idx,
        prox_idx,
        mid_idx,
        dist_idx,
        sim_time: 0.0,
        last_reset_time: 0.0,
    };

    let hand_mesh = meshes.add(Capsule3d::new(HAND_RADIUS as f32, (2.0 * HAND_HALF_H) as f32));
    let prox_mesh = meshes.add(Capsule3d::new(PROX_RADIUS as f32, (2.0 * PROX_HALF_H) as f32));
    let mid_mesh = meshes.add(Capsule3d::new(MID_RADIUS as f32, (2.0 * MID_HALF_H) as f32));
    let dist_mesh = meshes.add(Capsule3d::new(DIST_RADIUS as f32, (2.0 * DIST_HALF_H) as f32));

    let hand_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.78, 0.58, 0.45),
        ..default()
    });
    let phalanx_mat_a = materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.65, 0.50),
        ..default()
    });
    let phalanx_mat_b = materials.add(StandardMaterial {
        base_color: Color::srgb(0.90, 0.70, 0.55),
        ..default()
    });
    let phalanx_mat_c = materials.add(StandardMaterial {
        base_color: Color::srgb(0.95, 0.75, 0.60),
        ..default()
    });

    commands.spawn((
        PbrBundle {
            mesh: hand_mesh, material: hand_mat,
            transform: rigid_body_to_transform(&phys.world.bodies[hand_idx]),
            ..default()
        },
        BodyRef(hand_idx),
    ));
    commands.spawn((
        PbrBundle {
            mesh: prox_mesh, material: phalanx_mat_a,
            transform: rigid_body_to_transform(&phys.world.bodies[prox_idx]),
            ..default()
        },
        BodyRef(prox_idx),
    ));
    commands.spawn((
        PbrBundle {
            mesh: mid_mesh, material: phalanx_mat_b,
            transform: rigid_body_to_transform(&phys.world.bodies[mid_idx]),
            ..default()
        },
        BodyRef(mid_idx),
    ));
    commands.spawn((
        PbrBundle {
            mesh: dist_mesh, material: phalanx_mat_c,
            transform: rigid_body_to_transform(&phys.world.bodies[dist_idx]),
            ..default()
        },
        BodyRef(dist_idx),
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

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.95, 0.45)
            .looking_at(Vec3::new(0.0, 0.85, 0.0), Vec3::Y),
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
    if phys.sim_time - phys.last_reset_time >= CYCLE_PERIOD {
        reset_chain(&mut phys);
        let now = phys.sim_time;
        phys.last_reset_time = now;
    }

    let hand_idx = phys.hand_idx;
    for _ in 0..SOLVER_ITERATIONS {
        let body = &mut phys.world.bodies[hand_idx];
        let pos_err = body.position - HAND_REST_POS;
        let force = -ANCHOR_KP_LIN * pos_err - ANCHOR_KD_LIN * body.linear_velocity;
        let orient_err = body.orientation.scaled_axis();
        let torque = -ANCHOR_KP_ANG * orient_err - ANCHOR_KD_ANG * body.angular_velocity;
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
    mut query: Query<&mut Text, With<AngleReadout>>,
) {
    let h = &phys.world.bodies[phys.hand_idx];
    let p = &phys.world.bodies[phys.prox_idx];
    let m = &phys.world.bodies[phys.mid_idx];
    let d = &phys.world.bodies[phys.dist_idx];
    let mcp = (p.orientation * h.orientation.inverse()).scaled_axis().z;
    let pip = (m.orientation * p.orientation.inverse()).scaled_axis().z;
    let dip = (d.orientation * m.orientation.inverse()).scaled_axis().z;
    let elapsed = phys.sim_time - phys.last_reset_time;
    for mut text in &mut query {
        text.sections[0].value = format!(
            "passive finger relaxation\nt since reset: {:.2} s\n\nMCP: {:+.3} rad ({:+.1}°)\nPIP: {:+.3} rad ({:+.1}°)\nDIP: {:+.3} rad ({:+.1}°)\n\nresets every {:.0} s",
            elapsed,
            mcp, mcp.to_degrees(),
            pip, pip.to_degrees(),
            dip, dip.to_degrees(),
            CYCLE_PERIOD,
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Finger demo (passive biomechanics)".into(),
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
