// Quick visual demo of the swing-twist refactor: foot at its anatomical
// horizontal pose IN PHYSICS, rendered straight without any visual offset.
//
// Compare to `ankle_demo.rs` — that one has the foot vertical in physics
// and applies an R_x(-π/2) offset in the renderer to make it look
// horizontal. Here, the foot's body orientation IS the horizontal pose
// (baseline = R_x(-π/2)), so `rigid_body_to_transform(foot)` already
// gives the right rendering.
//
// PD controllers sweep through PF, then INV, then a simple combined
// motion. Watch the foot pivot at the ankle as if hinged on a real ankle
// joint — no rendering tricks.

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

const LOWER_LEG_HALF_H: f64 = 0.210;
const LOWER_LEG_RADIUS: f64 = 0.050;

const FOOT_MASS:    f64 = 1.0;
const FOOT_HALF_H:  f64 = 0.090;
const FOOT_RADIUS:  f64 = 0.032;
const FOOT_ANCHOR_LOCAL: NaVec3 = NaVec3::new(0.0, FOOT_HALF_H, 0.0);

const ANKLE_JOINT_K: f64 = 60_000.0;
const ANKLE_JOINT_C: f64 =    150.0;

const PF_K: f64 = 4.0;     const PF_C: f64 = 1.2;
const INV_K: f64 = 3.5;    const INV_C: f64 = 1.0;
const TWIST_K: f64 = 8.0;  const TWIST_C: f64 = 0.5;
const PF_MIN: f64 = -0.44;  const PF_MAX: f64 = 0.87;
const INV_MIN: f64 = -0.26; const INV_MAX: f64 = 0.52;
const TWIST_LIM: f64 = 0.10;
const ANKLE_LIMIT_K: f64 = 80.0;
const TWIST_LIMIT_K: f64 = 200.0;
const ANKLE_LIMIT_EXP: f64 = 2.0;

const ANKLE_PD_KP: f64 = 80.0;
const ANKLE_PD_KD: f64 = 6.0;

const SCHEDULE_PERIOD: f64 = 24.0;

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

fn half_cosine(t: f64, dur: f64, peak: f64) -> f64 {
    let p = (t / dur).clamp(0.0, 1.0);
    peak * 0.5 * (1.0 - (std::f64::consts::TAU * p).cos())
}

#[derive(Resource)]
struct PhysicsWorld {
    world: XpbdWorld,
    shin_idx: usize,
    foot_idx: usize,
    sim_time: f64,
    pd_pf: usize,
    pd_inv: usize,
    baseline: UnitQuaternion<f64>,
}

#[derive(Component)] struct BodyRef(usize);
#[derive(Component)] struct Readout;

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mut world = XpbdWorld::new();

    // Pinned shin, vertical.
    let mut shin = RigidBody::new(0.0, capsule_inertia(3.5, LOWER_LEG_RADIUS, LOWER_LEG_HALF_H));
    shin.position = NaVec3::new(0.0, 1.00, 0.0);
    shin.orientation = UnitQuaternion::identity();
    let shin_idx = world.add_body(shin);

    // Foot at HORIZONTAL physics pose: baseline = R_x(-π/2). Body +y points
    // world -z (heel side), body -y points world +z (toes forward).
    let baseline = UnitQuaternion::from_scaled_axis(
        NaVec3::new(-std::f64::consts::FRAC_PI_2, 0.0, 0.0));

    let mut foot = RigidBody::new(FOOT_MASS,
        capsule_inertia(FOOT_MASS, FOOT_RADIUS, FOOT_HALF_H));
    let (pos, orient) = place_child(
        &world.bodies[shin_idx],
        NaVec3::new(0.0, -LOWER_LEG_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL,
        baseline);
    foot.position = pos;
    foot.orientation = orient;
    let foot_idx = world.add_body(foot);

    add_elastic_joint(&mut world, shin_idx, foot_idx,
        NaVec3::new(0.0, -LOWER_LEG_HALF_H, 0.0),
        FOOT_ANCHOR_LOCAL,
        ANKLE_JOINT_K, ANKLE_JOINT_C);

    // Joint primitives use baseline so anatomical neutral reads as 0.
    let pf_axis    = NaVec3::new(1.0, 0.0, 0.0);
    let inv_axis   = NaVec3::new(0.0, 0.0, 1.0);
    let twist_axis = NaVec3::new(0.0, 1.0, 0.0);

    world.angular_soft_limits.push(
        AngularSoftLimit::new(shin_idx, foot_idx, pf_axis,
            PF_MIN, PF_MAX, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP).with_baseline(baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, pf_axis, PF_C));
    world.angular_springs.push(
        AngularSpring::new(shin_idx, foot_idx, pf_axis, PF_K, 0.0).with_baseline(baseline));

    world.angular_soft_limits.push(
        AngularSoftLimit::new(shin_idx, foot_idx, inv_axis,
            INV_MIN, INV_MAX, ANKLE_LIMIT_K, ANKLE_LIMIT_EXP).with_baseline(baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, inv_axis, INV_C));
    world.angular_springs.push(
        AngularSpring::new(shin_idx, foot_idx, inv_axis, INV_K, 0.0).with_baseline(baseline));

    world.angular_soft_limits.push(
        AngularSoftLimit::new(shin_idx, foot_idx, twist_axis,
            -TWIST_LIM, TWIST_LIM, TWIST_LIMIT_K, ANKLE_LIMIT_EXP).with_baseline(baseline));
    world.angular_dampers.push(AngularDamper::new(shin_idx, foot_idx, twist_axis, TWIST_C));
    world.angular_springs.push(
        AngularSpring::new(shin_idx, foot_idx, twist_axis, TWIST_K, 0.0).with_baseline(baseline));

    let pd_pf = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(shin_idx, foot_idx, pf_axis, ANKLE_PD_KP, ANKLE_PD_KD)
            .with_target(0.0).with_baseline(baseline));
    let pd_inv = world.angular_pd_controllers.len();
    world.angular_pd_controllers.push(
        AngularPDController::new(shin_idx, foot_idx, inv_axis, ANKLE_PD_KP, ANKLE_PD_KD)
            .with_target(0.0).with_baseline(baseline));

    let shin_mesh = meshes.add(Capsule3d::new(LOWER_LEG_RADIUS as f32, (2.0 * LOWER_LEG_HALF_H) as f32));
    let foot_mesh = meshes.add(Capsule3d::new(FOOT_RADIUS as f32, (2.0 * FOOT_HALF_H) as f32));
    let shin_mat = materials.add(StandardMaterial { base_color: Color::srgb(0.85, 0.65, 0.50), ..default() });
    let foot_mat = materials.add(StandardMaterial { base_color: Color::srgb(0.55, 0.35, 0.25), ..default() });

    commands.spawn((PbrBundle {
        mesh: shin_mesh, material: shin_mat,
        transform: rigid_body_to_transform(&world.bodies[shin_idx]),
        ..default()
    }, BodyRef(shin_idx)));
    // Foot rendered straight — its physics orientation IS its visual orientation.
    commands.spawn((PbrBundle {
        mesh: foot_mesh, material: foot_mat,
        transform: rigid_body_to_transform(&world.bodies[foot_idx]),
        ..default()
    }, BodyRef(foot_idx)));

    let ground = meshes.add(Plane3d::default().mesh().size(2.0, 2.0));
    let ground_mat = materials.add(StandardMaterial { base_color: Color::srgb(0.4, 0.45, 0.4), ..default() });
    commands.spawn(PbrBundle {
        mesh: ground, material: ground_mat,
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });

    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(1.10, 0.85, 0.50)
            .looking_at(Vec3::new(0.0, 0.78, 0.10), Vec3::Y),
        ..default()
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10_000.0, shadows_enabled: false, ..default() },
        transform: Transform::from_xyz(2.0, 4.0, 2.0)
            .looking_at(Vec3::new(0.0, 1.0, 0.0), Vec3::Y),
        ..default()
    });
    commands.insert_resource(AmbientLight { color: Color::WHITE, brightness: 220.0 });

    commands.spawn((TextBundle::from_section(
        "...",
        TextStyle { font_size: 18.0, color: Color::WHITE, ..default() },
    )
    .with_style(Style {
        position_type: PositionType::Absolute,
        top: Val::Px(12.0), left: Val::Px(12.0), ..default()
    }), Readout));

    commands.insert_resource(PhysicsWorld {
        world, shin_idx, foot_idx, sim_time: 0.0, pd_pf, pd_inv, baseline,
    });
}

fn ankle_targets(t: f64) -> (f64, f64, &'static str) {
    let t = t.rem_euclid(SCHEDULE_PERIOD);
    if t < 4.0 {
        (0.0, 0.0, "rest (foot at horizontal anatomical pose)")
    } else if t < 10.0 {
        let pf = half_cosine(t - 4.0, 6.0, 0.85);
        (pf, 0.0, "PF sweep (plantarflex / dorsiflex)")
    } else if t < 16.0 {
        let inv = half_cosine(t - 10.0, 6.0, 0.50);
        (0.0, inv, "INV sweep (inversion)")
    } else if t < 22.0 {
        // Combined PF + INV — the case where the OLD scaled_axis().dot()
        // would contaminate the locked twist axis. Now twist stays clean.
        let local = t - 16.0;
        let pf = half_cosine(local, 6.0, 0.50);
        let inv = half_cosine(local, 6.0, 0.30);
        (pf, inv, "combined PF + INV (twist stays clean — bivector fix)")
    } else {
        (0.0, 0.0, "relax to rest")
    }
}

fn physics_step(mut phys: ResMut<PhysicsWorld>) {
    let (pf_t, inv_t, _) = ankle_targets(phys.sim_time);
    let pd_pf = phys.pd_pf;
    let pd_inv = phys.pd_inv;
    phys.world.angular_pd_controllers[pd_pf].target_angle = pf_t;
    phys.world.angular_pd_controllers[pd_inv].target_angle = inv_t;
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
    let shin = &phys.world.bodies[phys.shin_idx];
    let foot = &phys.world.bodies[phys.foot_idx];
    let q_b_in_a = shin.orientation.inverse() * foot.orientation;
    let q_from_rest = q_b_in_a * phys.baseline.inverse();
    let pf    = physics::math::swing_twist_angle(q_from_rest, NaVec3::new(1.0, 0.0, 0.0));
    let inv   = physics::math::swing_twist_angle(q_from_rest, NaVec3::new(0.0, 0.0, 1.0));
    let twist = physics::math::swing_twist_angle(q_from_rest, NaVec3::new(0.0, 1.0, 0.0));
    let (pf_t, inv_t, phase) = ankle_targets(phys.sim_time);
    for mut text in &mut q {
        text.sections[0].value = format!(
            "{}\nt: {:.2} s\n\n  PF    : tgt {:+.2}  →  {:+.3} rad ({:+.1}°)\n  INV   : tgt {:+.2}  →  {:+.3} rad ({:+.1}°)\n  twist :              {:+.3} rad ({:+.1}°)\n\nFoot's physics orientation IS its anatomical horizontal pose.\nNo visual offset — `rigid_body_to_transform(foot)` is enough.",
            phase, phys.sim_time,
            pf_t, pf, pf.to_degrees(),
            inv_t, inv, inv.to_degrees(),
            twist, twist.to_degrees(),
        );
    }
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Ankle (horizontal foot in physics, no offset hack)".into(),
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
