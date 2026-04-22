use bevy::prelude::*;
use player::player::Player;
use player::segment::{BodyType, SegmentId};
use player::locomotion::Role;
use ball::football::Football;
use physics::math::Vec3;
use std::fs::File;
use std::io::{BufWriter, Write};

pub const DT: f64 = 1.0 / 120.0;

#[derive(Resource)]
pub struct CarrierRes {
    pub player: Player,
}

impl Default for CarrierRes {
    fn default() -> Self {
        let mut p = Player::new(BodyType::Receiver, Vec3::new(45.0, 0.0, 26.5));
        p.desired_velocity = Vec3::new(5.0, 0.0, 0.0);
        p.role = Role::RunningBack;
        Self { player: p }
    }
}

#[derive(Resource)]
pub struct DefenderRes {
    pub player: Player,
}

impl Default for DefenderRes {
    fn default() -> Self {
        let mut p = Player::new(BodyType::Lineman, Vec3::new(52.0, 0.0, 26.5));
        p.desired_velocity = Vec3::new(-5.0, 0.0, 0.0);
        p.role = Role::DefenderRush;
        Self { player: p }
    }
}

#[derive(Resource)]
pub struct BallRes {
    pub ball: Football,
}

impl Default for BallRes {
    fn default() -> Self {
        Self { ball: Football::new() }
    }
}

#[derive(Resource)]
pub struct SimConfig {
    pub dt: f64,
    pub elapsed: f64,
    pub log_tick: u64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self { dt: DT, elapsed: 0.0, log_tick: 0 }
    }
}

#[derive(Resource)]
pub struct SimLog(pub BufWriter<File>);

pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        let log_file = File::create("sim_log.csv").expect("cannot create sim_log.csv");
        let mut writer = BufWriter::new(log_file);
        writeln!(writer,
            "t,player,pelvis_x,pelvis_y,pelvis_z,vel_x,vel_y,vel_z,foot_l_y,foot_r_y,pelvis_lean_z,chest_lean_z,dist"
        ).unwrap();

        app
            .insert_resource(CarrierRes::default())
            .insert_resource(DefenderRes::default())
            .insert_resource(BallRes::default())
            .insert_resource(SimConfig::default())
            .insert_resource(SimLog(writer))
            .insert_resource(Time::<Fixed>::from_hz(120.0))
            .add_systems(FixedUpdate, step_physics);
    }
}

fn step_physics(
    mut carrier: ResMut<CarrierRes>,
    mut defender: ResMut<DefenderRes>,
    mut sim: ResMut<SimConfig>,
    mut log: ResMut<SimLog>,
) {
    let dt = sim.dt;
    sim.elapsed += dt;
    sim.log_tick += 1;

    let c_pel = &carrier.player.skeleton.bodies[SegmentId::Pelvis.idx()];
    let c_pos = c_pel.position;
    let c_vel = c_pel.linear_velocity;
    let c_pelvis_lean_z = c_pel.orientation.scaled_axis().z;
    let c_foot_l_y = carrier.player.skeleton.bodies[SegmentId::FootL.idx()].position.y;
    let c_foot_r_y = carrier.player.skeleton.bodies[SegmentId::FootR.idx()].position.y;
    let c_chest_lean_z = carrier.player.skeleton.bodies[SegmentId::Chest.idx()].orientation.scaled_axis().z;

    let d_pel = &defender.player.skeleton.bodies[SegmentId::Pelvis.idx()];
    let d_pos = d_pel.position;
    let d_vel = d_pel.linear_velocity;
    let d_pelvis_lean_z = d_pel.orientation.scaled_axis().z;
    let d_foot_l_y = defender.player.skeleton.bodies[SegmentId::FootL.idx()].position.y;
    let d_foot_r_y = defender.player.skeleton.bodies[SegmentId::FootR.idx()].position.y;
    let d_chest_lean_z = defender.player.skeleton.bodies[SegmentId::Chest.idx()].orientation.scaled_axis().z;

    let dist = (c_pos - d_pos).norm();

    writeln!(log.0,
        "{:.4},carrier,{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
        sim.elapsed, c_pos.x, c_pos.y, c_pos.z, c_vel.x, c_vel.y, c_vel.z,
        c_foot_l_y, c_foot_r_y, c_pelvis_lean_z, c_chest_lean_z, dist
    ).unwrap();
    writeln!(log.0,
        "{:.4},defender,{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
        sim.elapsed, d_pos.x, d_pos.y, d_pos.z, d_vel.x, d_vel.y, d_vel.z,
        d_foot_l_y, d_foot_r_y, d_pelvis_lean_z, d_chest_lean_z, dist
    ).unwrap();

    if dist < 0.8 {
        // Tackle state: kill drive, disable gravity cancel, lower pelvis target.
        // Without gravity cancel the carrier falls under real gravity (~9.81 m/s²).
        // The lowered pelvis spring then adds an additional ~50 m/s² downward pull.
        carrier.player.velocity_drive_scale = 0.0;
        carrier.player.gravity_cancel_active = false;
        carrier.player.pelvis_height_target = 0.15;

        let contact_dir = if dist > 1e-6 {
            (d_pos - c_pos).normalize()
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };

        // Impulse at chest (above pelvis COM) creates angular momentum that topples
        // the carrier forward rather than just sliding them back.
        let c_chest_pos = carrier.player.skeleton.bodies[SegmentId::Chest.idx()].position;
        let topple = (-contact_dir * 120.0 + Vec3::new(0.0, -80.0, 0.0)) * dt;
        carrier.player.skeleton.bodies[SegmentId::Chest.idx()]
            .apply_impulse(topple, c_chest_pos);

        // Defender absorbs a push-back impulse.
        defender.player.skeleton.bodies[SegmentId::Pelvis.idx()]
            .apply_impulse(contact_dir * 60.0 * dt, d_pos);
    }

    carrier.player.step(dt);
    defender.player.step(dt);
}
