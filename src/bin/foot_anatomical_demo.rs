// Visual demo of the 26-bone anatomical foot (Bevy).
//
// Default mode: bare-foot showcase. The world is `foot::build_foot` alone
// (no shin, no load) dropped from ~6 cm — the same configuration as the
// foot module's settle test, which is stable indefinitely. After a 1 s
// settle the demo cycles through commanded-motion scenarios (~4 s each,
// targets return to 0 between scenarios):
//   1. Neutral stance (manifold contact visible)
//   2. Toe push-off (MTP dorsiflexion)
//   3. Toe curl (MTP + IP/PIP/DIP flexion)
//   4. Subtalar inversion (+0.35 rad about the oblique Manter/Inman axis —
//      the ankle-sprain mechanism axis)
//   5. Subtalar eversion (-0.15 rad)
//
// --loaded: honest full-load mode. Mirrors `ankle_anatomical.rs`'s
// build_world (shin + talocrural + 7 kg stick ≈ 11 kg total) and lets it
// run. It stands briefly, then twist-collapses through the compliant
// subtalar/midfoot joints — the documented current limit: arch rigidity
// (plantar fascia / windlass) is Phase 3 and not built yet.
//
// Contact visualization: every frame, `world.engaged_endpoints` is queried
// across all bodies and a bright emissive marker is drawn at each active
// manifold contact point, plus a red marker at the force-weighted center
// of pressure. The migrating contact pattern during the scenario cycle is
// the visual proof of the per-substep manifold contact.
//
// Env: FOOT_DEMO_EXIT_AFTER=<seconds> auto-exits (used by headless checks).

use bevy::app::AppExit;
use bevy::prelude::*;

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::math::Vec3 as NaVec3;

use improper_football::anatomy::foot::{self, Bone, FootHandles, JointKind, BONES, BONE_COUNT, JOINTS};
use improper_football::render_util::{nalgebra_vec3_to_bevy, rigid_body_to_transform};

// ─── Simulation timing (identical to the validation binaries) ──────────

const PHYSICS_HZ: f64 = 120.0;
const DT: f64 = 1.0 / PHYSICS_HZ;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;

const SETTLE_TIME: f64 = 1.0;
const SCENARIO_TIME: f64 = 5.0;

// ─── Loaded-mode body parameters (mirror ankle_anatomical.rs) ───────────

const SHIN_MASS: f64 = 3.00;
const SHIN_HALF_H: f64 = 0.21;
const SHIN_RADIUS: f64 = 0.05;

const STICK_MASS: f64 = 7.00; // load above shin; total above-ankle ≈ 10 kg
const STICK_HALF_H: f64 = 0.40;
const STICK_RADIUS: f64 = 0.08;

// ─── Modes & scenarios ──────────────────────────────────────────────────

#[derive(Resource, Clone, Copy, PartialEq)]
enum Mode {
    Bare,
    Loaded,
}

const SCENARIO_COUNT: usize = 5;

fn scenario_name(i: usize) -> &'static str {
    match i {
        0 => "Neutral stance",
        1 => "Toe push-off",
        2 => "Toe curl",
        3 => "Subtalar inversion (sprain axis)",
        4 => "Subtalar eversion",
        _ => unreachable!(),
    }
}

/// The four lesser-toe chains: [proximal, middle, distal] phalanx.
const LESSER_TOES: [[Bone; 3]; 4] = [
    [Bone::Toe2Pp, Bone::Toe2Mp, Bone::Toe2Dp],
    [Bone::Toe3Pp, Bone::Toe3Mp, Bone::Toe3Dp],
    [Bone::Toe4Pp, Bone::Toe4Mp, Bone::Toe4Dp],
    [Bone::Toe5Pp, Bone::Toe5Mp, Bone::Toe5Dp],
];

/// ROM of the articulated joint whose CHILD is `child`, from the JOINTS
/// table (the subtalar axis override doesn't change its ROM entry).
fn joint_rom(child: Bone) -> Option<(f64, f64)> {
    JOINTS.iter().find_map(|s| match s.kind {
        JointKind::Articulated { rom_min, rom_max, .. } if s.child as usize == child as usize => {
            Some((rom_min, rom_max))
        }
        _ => None,
    })
}

/// Rate at which PD targets ramp toward their scenario goals (rad/s).
/// Step commands are NOT used: a 0.8 rad step through a kp=30 PD is a
/// ~24 N·m transient on a 1 kg free-standing foot (whose gravity torques
/// are ~1 N·m) and launches it airborne. Even ramped, the reaction torque
/// on the foot is ≈ (passive damper c + PD lag) × rate summed over the
/// commanded joints, so the rate must keep that total well under ~1 N·m
/// (the bare foot's gravity-restoring torque about its forefoot).
const TARGET_RAMP_RATE: f64 = 0.2;

/// Smoothly-ramped PD goals for the 21 articulated foot joints, keyed by
/// child bone. Goals are set per scenario; `tick` moves each PD's actual
/// `target_angle` toward its goal at TARGET_RAMP_RATE.
struct Commander {
    goals: [f64; BONE_COUNT],
}

impl Commander {
    fn new() -> Self {
        Commander { goals: [0.0; BONE_COUNT] }
    }

    /// Set the goal for the joint whose child is `child`, clamped to the
    /// JOINTS-table ROM (commands must never exceed the biological range).
    fn set(&mut self, child: Bone, goal: f64) {
        let clamped = match joint_rom(child) {
            Some((lo, hi)) => goal.clamp(lo, hi),
            None => goal,
        };
        self.goals[child as usize] = clamped;
    }

    /// How long into a scenario its goals are held before ramping back to
    /// neutral. Every scenario is a pulse — out, hold, back — so the foot
    /// returns to neutral stance before the next scenario, and any slow
    /// roll a held command induces never integrates to a topple. The
    /// subtalar scenarios return earlier (shorter hold) because they sit
    /// closest to the free foot's static tipping threshold.
    fn hold_time(i: usize) -> f64 {
        match i {
            3 | 4 => 2.0,
            _ => 3.0,
        }
    }

    /// Per-frame goal refresh: scenario goals while inside the hold window,
    /// neutral afterwards (the ramp in `tick` does the smoothing).
    fn update(&mut self, i: usize, scenario_t: f64) {
        if scenario_t < Self::hold_time(i) {
            self.set_scenario(i);
        } else {
            self.goals = [0.0; BONE_COUNT];
        }
    }

    /// Zero every goal, then apply scenario `i`'s commands.
    fn set_scenario(&mut self, i: usize) {
        self.goals = [0.0; BONE_COUNT];
        match i {
            0 => {} // neutral stance: no commands
            1 => {
                // Toe push-off: MTP dorsiflexion (hallux ROM [-0.61, 1.22],
                // lesser ±0.70). Amplitudes REDUCED from the validation
                // test's +0.8/+0.5: probed headlessly, the bare 1 kg foot
                // (nothing anchoring it; gravity-restoring torque < 1 N·m)
                // is flipped by the hallux MTP's kp=30 at +0.3 and above
                // even with the extra damping; +0.1 hallux / +0.4 lesser
                // is stable. The loaded validation world (11 kg column)
                // tracks the full +0.8 fine (test_mtp_pushoff).
                self.set(Bone::HalluxPp, 0.1);
                for chain in LESSER_TOES {
                    self.set(chain[0], 0.4);
                }
            }
            2 => {
                // Toe curl: flexion at the toe joints (within the ROM
                // mins: MTP -0.61/-0.70, IP -1.57, PIP -0.61, DIP -1.05).
                // The hallux is curled through its IP (kp=6) rather than
                // its strong MTP, for the same stability reason as above.
                self.set(Bone::HalluxDp, -0.4);
                for chain in LESSER_TOES {
                    self.set(chain[0], -0.2);
                    self.set(chain[1], -0.4);
                    self.set(chain[2], -0.4);
                }
            }
            3 => {
                // Subtalar inversion about the oblique Manter/Inman axis
                // (42° inclination, 23° medial deviation — the sprain
                // mechanism axis). ROM is [-0.17, 0.44] but the usable
                // amplitude is bounded by STATICS, not ROM: the free
                // foot's tipping threshold is only m·g·(edge margin)
                // ≈ 0.25 N·m, so a held inversion of 0.07 rad or more
                // rolls it off its lateral edge (probed) — the sprain
                // mechanism itself. ±0.05 rad shifts the contact pattern
                // and CoP visibly and stays up.
                self.set(Bone::Calcaneus, 0.05);
            }
            4 => {
                // Eversion: mirror of scenario 3 (ROM min -0.17).
                self.set(Bone::Calcaneus, -0.05);
            }
            _ => unreachable!(),
        }
    }

    /// Move every PD target one frame-step toward its goal.
    fn tick(&self, world: &mut XpbdWorld, foot: &FootHandles, dt: f64) {
        let max_step = TARGET_RAMP_RATE * dt;
        for child in 0..BONE_COUNT {
            let pd_idx = foot.pd_by_child[child];
            if pd_idx == usize::MAX {
                continue; // welded child — no PD
            }
            let pd = &mut world.angular_pd_controllers[pd_idx];
            let delta = self.goals[child] - pd.target_angle;
            pd.target_angle += delta.clamp(-max_step, max_step);
        }
    }
}

// ─── World construction ─────────────────────────────────────────────────

struct Built {
    world: XpbdWorld,
    foot: FootHandles,
    /// Loaded mode only: (shin_idx, stick_idx).
    extra: Option<(usize, usize)>,
}

/// Bare showcase: the foot alone, starting at its resting contact height
/// (attach.y = 0.039 puts the flat sole on y=0 — the same bare-foot
/// configuration as the validation suite's MTP push-off test).
///
/// NOTE: a ~6 cm drop (the settle test's start) is NOT used here: probed
/// headlessly, the bounce rolls the unsupported foot ~130° onto its
/// medial side — the settle test only asserts no explosion, not an
/// upright landing. From resting height the bare foot stands upright
/// through the whole scenario cycle.
fn build_bare() -> Built {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let foot = foot::build_foot(&mut world, NaVec3::new(0.0, 0.039, 0.0));

    // Extra passive damping, host-side composition with the engine's
    // AngularDamper (as attach_talocrural and the balance demos do).
    // Probed finding: on the UNLOADED 1 kg foot, commanding the strong
    // PDs (hallux MTP kp=30, subtalar kp=50) against the ground manifold
    // excites a slowly-growing oscillation that detonates the foot
    // airborne within ~1-3 s — even AFTER the target has returned to 0
    // (the ring keeps growing). The validated loaded world never shows
    // this; its 11 kg column anchors the foot. The added joint damping
    // drains the ring so the commanded scenarios can run bare.
    let subtalar_axis = foot::anat_to_world(foot::subtalar_axis_anat());
    world.angular_dampers.push(physics::angular_damper::AngularDamper::new(
        foot.body(Bone::Talus), foot.body(Bone::Calcaneus), subtalar_axis, 3.0));
    for spec in foot::JOINTS.iter() {
        if let JointKind::Articulated { axis_anat, .. } = spec.kind {
            let axis = if spec.parent == Bone::Talus && spec.child == Bone::Calcaneus {
                subtalar_axis
            } else {
                foot::anat_to_world(axis_anat)
            };
            world.angular_dampers.push(physics::angular_damper::AngularDamper::new(
                foot.body(spec.parent), foot.body(spec.child), axis, 1.0));
        }
    }

    Built { world, foot, extra: None }
}

/// Full-load mode: mirrors ankle_anatomical.rs's build_world — shin +
/// talocrural joint + 7 kg stick welded at the shin top.
fn build_loaded() -> Built {
    let mut world = XpbdWorld::new();
    world.set_ground(0.0);
    let id = nalgebra::UnitQuaternion::identity();

    // attach.y = 0.039 puts the flat sole on y=0 (see ankle_anatomical.rs).
    let attach = NaVec3::new(0.0, 0.039, 0.0);

    let mut shin = RigidBody::new_capsule(SHIN_MASS, SHIN_HALF_H, SHIN_RADIUS);
    shin.position = NaVec3::new(0.0, attach.y + SHIN_HALF_H, 0.0);
    shin.orientation = id;
    let shin_idx = world.add_body(shin);

    let foot = foot::build_foot(&mut world, attach);

    foot::attach_talocrural(&mut world, shin_idx, &foot, -0.35, 0.70, 800.0, 40.0);

    let mut stick = RigidBody::new_capsule(STICK_MASS, STICK_HALF_H, STICK_RADIUS);
    stick.position = NaVec3::new(0.0, attach.y + 2.0 * SHIN_HALF_H + STICK_HALF_H, 0.0);
    stick.orientation = id;
    let stick_idx = world.add_body(stick);
    let weld_pt = NaVec3::new(0.0, attach.y + 2.0 * SHIN_HALF_H, 0.0); // shin top
    let anchor_shin = foot::anchor_at(&world, shin_idx, weld_pt);
    let anchor_stick = foot::anchor_at(&world, stick_idx, weld_pt);
    foot::weld(&mut world, shin_idx, stick_idx, anchor_shin, anchor_stick);

    Built { world, foot, extra: Some((shin_idx, stick_idx)) }
}

// ─── Bevy resources & components ────────────────────────────────────────

#[derive(Resource)]
struct Sim {
    world: XpbdWorld,
    foot: FootHandles,
    mode: Mode,
    /// Loaded mode: (shin_idx, stick_idx).
    extra: Option<(usize, usize)>,
    sim_time: f64,
    /// None while settling (bare mode); Some(i) once the cycle has begun.
    scenario: Option<usize>,
    scenario_t: f64,
    commander: Commander,
    /// Active manifold contacts this frame: (world point, normal force N).
    contacts: Vec<(NaVec3, f64)>,
    /// Force-weighted center of pressure.
    cop: Option<NaVec3>,
    fn_total: f64,
    /// Loaded mode: running max tilt of the load stick (degrees).
    max_tilt_deg: f64,
    next_tilt_log: f64,
    exit_after: Option<f64>,
}

#[derive(Component)]
struct BodyRef(usize);

#[derive(Component)]
struct ContactMarker(usize);

#[derive(Component)]
struct CopMarker;

#[derive(Component)]
struct Readout;

// ─── Setup ──────────────────────────────────────────────────────────────

fn setup(
    mode: Res<Mode>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let built = match *mode {
        Mode::Bare => build_bare(),
        Mode::Loaded => build_loaded(),
    };
    let world = built.world;
    let foot = built.foot;

    // Bone materials by anatomical group.
    let tarsal_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.93, 0.89, 0.80), // warm bone-white
        perceptual_roughness: 0.55,
        ..default()
    });
    let metatarsal_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.86, 0.75, 0.58), // light tan
        perceptual_roughness: 0.6,
        ..default()
    });
    let phalanx_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.76, 0.62, 0.47), // slightly darker
        perceptual_roughness: 0.6,
        ..default()
    });

    // One capsule mesh per bone (each bone has its own dimensions).
    for i in 0..BONE_COUNT {
        let p = &BONES[i];
        let mesh = meshes.add(Capsule3d::new(p.radius as f32, (2.0 * p.half_h) as f32));
        let mat = match i {
            0..=6 => tarsal_mat.clone(),       // calcaneus..cuneiforms
            7..=11 => metatarsal_mat.clone(),  // mt1..mt5
            _ => phalanx_mat.clone(),          // phalanges
        };
        commands.spawn((
            PbrBundle {
                mesh,
                material: mat,
                transform: rigid_body_to_transform(&world.bodies[foot.bodies[i]]),
                ..default()
            },
            BodyRef(foot.bodies[i]),
        ));
    }

    // Loaded mode: shin + stick in neutral gray.
    if let Some((shin_idx, stick_idx)) = built.extra {
        let gray = materials.add(StandardMaterial {
            base_color: Color::srgb(0.55, 0.55, 0.58),
            perceptual_roughness: 0.7,
            ..default()
        });
        let shin_mesh = meshes.add(Capsule3d::new(SHIN_RADIUS as f32, (2.0 * SHIN_HALF_H) as f32));
        let stick_mesh = meshes.add(Capsule3d::new(STICK_RADIUS as f32, (2.0 * STICK_HALF_H) as f32));
        commands.spawn((
            PbrBundle {
                mesh: shin_mesh,
                material: gray.clone(),
                transform: rigid_body_to_transform(&world.bodies[shin_idx]),
                ..default()
            },
            BodyRef(shin_idx),
        ));
        commands.spawn((
            PbrBundle {
                mesh: stick_mesh,
                material: gray,
                transform: rigid_body_to_transform(&world.bodies[stick_idx]),
                ..default()
            },
            BodyRef(stick_idx),
        ));
    }

    // Contact markers: a pooled set (2 capsule endpoints × every body),
    // repositioned/shown each frame. Bright emissive so they read as the
    // "money shot" proof of the manifold contact.
    let marker_mesh = meshes.add(Sphere::new(0.0035));
    let marker_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.85, 0.1),
        emissive: LinearRgba::rgb(3.0, 2.2, 0.2),
        ..default()
    });
    let n_bodies = world.bodies.len();
    for slot in 0..(2 * n_bodies) {
        commands.spawn((
            PbrBundle {
                mesh: marker_mesh.clone(),
                material: marker_mat.clone(),
                visibility: Visibility::Hidden,
                ..default()
            },
            ContactMarker(slot),
        ));
    }
    // Center-of-pressure marker (force-weighted centroid), red.
    let cop_mesh = meshes.add(Sphere::new(0.006));
    let cop_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.2, 0.2),
        emissive: LinearRgba::rgb(3.0, 0.3, 0.3),
        ..default()
    });
    commands.spawn((
        PbrBundle {
            mesh: cop_mesh,
            material: cop_mat,
            visibility: Visibility::Hidden,
            ..default()
        },
        CopMarker,
    ));

    // Ground plane.
    let ground_mesh = meshes.add(Plane3d::default().mesh().size(2.5, 2.5));
    let ground_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.42, 0.46, 0.42),
        perceptual_roughness: 0.95,
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: ground_mesh,
        material: ground_material,
        transform: Transform::from_xyz(0.0, 0.0, 0.0),
        ..default()
    });

    // Camera: three-quarter anterior-lateral view. The foot is ~25 cm
    // long, so the bare-mode camera sits close; loaded mode pulls back to
    // frame the 1.3 m column and its collapse.
    let camera_transform = match *mode {
        Mode::Bare => Transform::from_xyz(0.45, 0.30, 0.50)
            .looking_at(Vec3::new(0.0, 0.05, 0.08), Vec3::Y),
        Mode::Loaded => Transform::from_xyz(1.1, 0.85, 1.3)
            .looking_at(Vec3::new(0.0, 0.45, 0.0), Vec3::Y),
    };
    commands.spawn(Camera3dBundle {
        transform: camera_transform,
        ..default()
    });

    // Soft directional key light + ambient fill so bone colors read.
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 9_000.0,
            shadows_enabled: true,
            ..default()
        },
        transform: Transform::from_xyz(1.5, 3.0, 1.0)
            .looking_at(Vec3::new(0.0, 0.0, 0.05), Vec3::Y),
        ..default()
    });
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
    });

    // UI overlay: solid dark panel behind white text — legible on both
    // light and dark backgrounds.
    commands.spawn((
        TextBundle::from_section(
            "...",
            TextStyle {
                font_size: 17.0,
                color: Color::WHITE,
                ..default()
            },
        )
        .with_style(Style {
            position_type: PositionType::Absolute,
            top: Val::Px(12.0),
            left: Val::Px(12.0),
            padding: UiRect::all(Val::Px(8.0)),
            ..default()
        })
        .with_background_color(Color::srgba(0.0, 0.0, 0.0, 0.65)),
        Readout,
    ));

    let exit_after = std::env::var("FOOT_DEMO_EXIT_AFTER")
        .ok()
        .and_then(|s| s.parse::<f64>().ok());

    commands.insert_resource(Sim {
        world,
        foot,
        mode: *mode,
        extra: built.extra,
        sim_time: 0.0,
        scenario: None,
        scenario_t: 0.0,
        commander: Commander::new(),
        contacts: Vec::new(),
        cop: None,
        fn_total: 0.0,
        max_tilt_deg: 0.0,
        next_tilt_log: 1.0,
        exit_after,
    });
}

// ─── Physics step + scenario state machine ──────────────────────────────

fn gather_contacts(world: &XpbdWorld) -> (Vec<(NaVec3, f64)>, Option<NaVec3>, f64) {
    let mut contacts = Vec::new();
    let mut num = NaVec3::zeros();
    let mut den = 0.0;
    for idx in 0..world.bodies.len() {
        for e in world.engaged_endpoints(idx).iter().flatten() {
            contacts.push((e.contact_point, e.normal_force));
            num += e.contact_point * e.normal_force;
            den += e.normal_force;
        }
    }
    let cop = if den > 1e-6 { Some(num / den) } else { None };
    (contacts, cop, den)
}

fn physics_step(mut sim: ResMut<Sim>, mut exit: EventWriter<AppExit>) {
    {
        // Refresh scenario goals (hold window, then neutral) and ramp PD
        // targets toward them (split-borrow out of the resource).
        let Sim { world, foot, commander, mode, scenario, scenario_t, .. } = &mut *sim;
        if *mode == Mode::Bare {
            if let Some(i) = *scenario {
                commander.update(i, *scenario_t);
            }
            commander.tick(world, foot, DT);
        }
    }
    for _ in 0..SOLVER_ITERATIONS {
        sim.world.substep(SUB_DT);
    }
    sim.sim_time += DT;

    let (contacts, cop, fn_total) = gather_contacts(&sim.world);
    sim.contacts = contacts;
    sim.cop = cop;
    sim.fn_total = fn_total;

    match sim.mode {
        Mode::Bare => {
            // Scenario cycle: settle, then SCENARIO_TIME per scenario, loop.
            let next = match sim.scenario {
                None if sim.sim_time >= SETTLE_TIME => Some(0),
                Some(i) => {
                    sim.scenario_t += DT;
                    if sim.scenario_t >= SCENARIO_TIME {
                        Some((i + 1) % SCENARIO_COUNT)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(i) = next {
                sim.scenario = Some(i);
                sim.scenario_t = 0.0;
                println!(
                    "t={:6.2}s  -> scenario {}/{}: {:<34} (contacts={}, Fn={:.1} N)",
                    sim.sim_time,
                    i + 1,
                    SCENARIO_COUNT,
                    scenario_name(i),
                    sim.contacts.len(),
                    sim.fn_total,
                );
            }
        }
        Mode::Loaded => {
            // No commands — just watch, and log the tilt of the load stick.
            if let Some((_, stick_idx)) = sim.extra {
                let tilt = sim.world.bodies[stick_idx]
                    .orientation
                    .scaled_axis()
                    .norm()
                    .to_degrees();
                if tilt > sim.max_tilt_deg {
                    sim.max_tilt_deg = tilt;
                }
                if sim.sim_time >= sim.next_tilt_log {
                    sim.next_tilt_log += 1.0;
                    println!(
                        "t={:6.2}s  stick tilt={:6.1} deg (max {:6.1})  contacts={}  Fn={:.1} N",
                        sim.sim_time,
                        tilt,
                        sim.max_tilt_deg,
                        sim.contacts.len(),
                        sim.fn_total,
                    );
                }
            }
        }
    }

    if let Some(limit) = sim.exit_after {
        if sim.sim_time >= limit {
            if sim.mode == Mode::Loaded {
                println!("exit: max stick tilt reached {:.1} deg", sim.max_tilt_deg);
            }
            exit.send(AppExit::Success);
        }
    }
}

// ─── Render sync ────────────────────────────────────────────────────────

fn sync_transforms(sim: Res<Sim>, mut q: Query<(&BodyRef, &mut Transform)>) {
    for (body_ref, mut t) in &mut q {
        *t = rigid_body_to_transform(&sim.world.bodies[body_ref.0]);
    }
}

fn sync_contact_markers(
    sim: Res<Sim>,
    mut markers: Query<(&ContactMarker, &mut Transform, &mut Visibility), Without<CopMarker>>,
    mut cop_q: Query<(&mut Transform, &mut Visibility), With<CopMarker>>,
) {
    for (marker, mut t, mut vis) in &mut markers {
        if let Some(&(p, _)) = sim.contacts.get(marker.0) {
            t.translation = nalgebra_vec3_to_bevy(p);
            *vis = Visibility::Visible;
        } else {
            *vis = Visibility::Hidden;
        }
    }
    for (mut t, mut vis) in &mut cop_q {
        match sim.cop {
            Some(c) => {
                t.translation = nalgebra_vec3_to_bevy(c);
                *vis = Visibility::Visible;
            }
            None => *vis = Visibility::Hidden,
        }
    }
}

fn update_readout(sim: Res<Sim>, mut q: Query<&mut Text, With<Readout>>) {
    let value = match sim.mode {
        Mode::Bare => {
            let scenario = match sim.scenario {
                None => "settling...".to_string(),
                Some(i) => format!("{} ({}/{})", scenario_name(i), i + 1, SCENARIO_COUNT),
            };
            format!(
                "26-bone anatomical foot — manifold ground contact\n\
                 scenario: {}\n\
                 sim time: {:.1} s\n\
                 contacts: {}   total Fn: {:.1} N",
                scenario,
                sim.sim_time,
                sim.contacts.len(),
                sim.fn_total,
            )
        }
        Mode::Loaded => format!(
            "full 11 kg load: articulated foot twists off support — \
             arch rigidity awaits Phase 3 windlass\n\
             sim time: {:.1} s   stick tilt (max): {:.1} deg\n\
             contacts: {}   total Fn: {:.1} N",
            sim.sim_time,
            sim.max_tilt_deg,
            sim.contacts.len(),
            sim.fn_total,
        ),
    };
    for mut text in &mut q {
        text.sections[0].value = value.clone();
    }
}

// ─── Headless probe (--probe): same world + scenario cycle, no window ───
//
// Prints pose/contact telemetry every 0.5 s so the scenario cycle can be
// verified (and visually tuned) without rendering.

fn probe() {
    let Built { mut world, foot, .. } = build_bare();
    let baseline = world.bodies[foot.body(Bone::Talus)].orientation;
    let mut commander = Commander::new();
    let mut scenario: Option<usize> = None;
    let mut scenario_t = 0.0;
    let total = std::env::var("FOOT_PROBE_TOTAL")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(SETTLE_TIME + 2.0 * SCENARIO_TIME * SCENARIO_COUNT as f64);
    let interval = std::env::var("FOOT_PROBE_INTERVAL")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.5);
    let start_scenario = std::env::var("FOOT_PROBE_START")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0)
        % SCENARIO_COUNT;
    let mut t = 0.0;
    let mut next_print = 0.0;
    while t < total {
        if let Some(i) = scenario {
            commander.update(i, scenario_t);
        }
        commander.tick(&mut world, &foot, DT);
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;
        match scenario {
            None if t >= SETTLE_TIME => {
                scenario = Some(start_scenario);
                scenario_t = 0.0;
            }
            Some(i) => {
                scenario_t += DT;
                if scenario_t >= SCENARIO_TIME {
                    scenario = Some((i + 1) % SCENARIO_COUNT);
                    scenario_t = 0.0;
                }
            }
            _ => {}
        }
        if t >= next_print {
            next_print += interval;
            let (contacts, cop, fn_total) = gather_contacts(&world);
            let calc = &world.bodies[foot.body(Bone::Calcaneus)];
            let talus = &world.bodies[foot.body(Bone::Talus)];
            let tilt = (baseline.inverse() * calc.orientation).scaled_axis().norm().to_degrees();
            let hallux = world.bodies[foot.body(Bone::HalluxDp)].position;
            let (mut ymin, mut ymax) = (f64::MAX, f64::MIN);
            for i in 0..BONE_COUNT {
                let y = world.bodies[foot.bodies[i]].position.y;
                ymin = ymin.min(y);
                ymax = ymax.max(y);
            }
            println!(
                "t={:5.1}s [{}] contacts={:2} Fn={:6.2}N calc-tilt={:6.1}deg talus=({:+.3},{:.3},{:+.3}) hallux_dp=({:+.3},{:.3},{:+.3}) y[{:.3},{:.3}] cop_z={}",
                t,
                scenario.map_or("settle".into(), |i| format!("{}:{}", i + 1, scenario_name(i))),
                contacts.len(),
                fn_total,
                tilt,
                talus.position.x, talus.position.y, talus.position.z,
                hallux.x, hallux.y, hallux.z,
                ymin, ymax,
                cop.map_or("none".into(), |c| format!("{:+.3}", c.z)),
            );
        }
    }
}

// ─── Main ───────────────────────────────────────────────────────────────

fn main() {
    if std::env::args().any(|a| a == "--probe") {
        probe();
        return;
    }
    let loaded = std::env::args().any(|a| a == "--loaded");
    let mode = if loaded { Mode::Loaded } else { Mode::Bare };
    let title = match mode {
        Mode::Bare => "26-bone anatomical foot — manifold contact showcase",
        Mode::Loaded => "26-bone anatomical foot — full-load limit (--loaded)",
    };
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: title.into(),
                resolution: (1100.0, 800.0).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(mode)
        .insert_resource(Time::<Fixed>::from_hz(PHYSICS_HZ))
        .add_systems(Startup, setup)
        .add_systems(FixedUpdate, physics_step)
        .add_systems(Update, (sync_transforms, sync_contact_markers, update_readout))
        .run();
}
