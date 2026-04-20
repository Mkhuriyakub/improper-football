use bevy::prelude::*;
use player::player::Player;
use player::segment::{BodyType, SegmentId};
use player::locomotion::Role;
use ball::football::Football;
use physics::math::Vec3;

pub const DT: f64 = 1.0 / 120.0;

#[derive(Resource)]
pub struct CarrierRes {
    pub player: Player,
}

impl Default for CarrierRes {
    fn default() -> Self {
        let mut p = Player::new(BodyType::Receiver, Vec3::new(0.0, 0.0, 0.0));
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
        let mut p = Player::new(BodyType::Lineman, Vec3::new(4.0, 0.0, 0.0));
        p.desired_velocity = Vec3::zeros();
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
}

impl Default for SimConfig {
    fn default() -> Self {
        Self { dt: DT, elapsed: 0.0 }
    }
}

pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app
            .insert_resource(CarrierRes::default())
            .insert_resource(DefenderRes::default())
            .insert_resource(BallRes::default())
            .insert_resource(SimConfig::default())
            .insert_resource(Time::<Fixed>::from_hz(120.0))
            .add_systems(FixedUpdate, step_physics);
    }
}

fn step_physics(
    mut carrier: ResMut<CarrierRes>,
    mut defender: ResMut<DefenderRes>,
    mut sim: ResMut<SimConfig>,
) {
    let dt = sim.dt;
    sim.elapsed += dt;

    let c_pos = carrier.player.skeleton.bodies[SegmentId::Pelvis.idx()].position;
    let d_pos = defender.player.skeleton.bodies[SegmentId::Pelvis.idx()].position;
    let dist = (c_pos - d_pos).norm();

    if dist < 0.8 {
        let contact_dir = if dist > 1e-6 {
            (d_pos - c_pos).normalize()
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };
        let impulse_mag = 600.0 * dt;
        carrier.player.skeleton.bodies[SegmentId::Pelvis.idx()]
            .apply_impulse(-contact_dir * impulse_mag, c_pos);
        defender.player.skeleton.bodies[SegmentId::Pelvis.idx()]
            .apply_impulse(contact_dir * impulse_mag, d_pos);
        carrier.player.skeleton.bodies[SegmentId::Chest.idx()]
            .apply_force(Vec3::new(0.0, -2000.0, 0.0));
    }

    carrier.player.step(dt);
    defender.player.step(dt);
}
