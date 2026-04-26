use bevy::prelude::*;
use crate::plugins::sim::CarrierRes;
use crate::render_util::nalgebra_vec3_to_bevy;
use player::segment::SegmentId;

#[derive(Resource)]
pub struct CameraConfig {
    pub offset: Vec3,
    pub alpha: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            // Close-up: 1.8m in front of the player at chest height.
            // Change to Vec3::new(-8.0, 6.0, 8.0) for the full-field view.
            offset: Vec3::new(0.0, 0.42, 1.8),
            alpha: 0.15,
        }
    }
}

pub fn spring_follow(current: Vec3, target: Vec3, alpha: f32) -> Vec3 {
    current.lerp(target, alpha)
}

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CameraConfig::default())
           .add_systems(Startup, spawn_camera)
           .add_systems(Update, follow_action);
    }
}

fn spawn_camera(mut commands: Commands) {
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(37.0, 8.0, 34.5)
            .looking_at(Vec3::new(48.5, 1.0, 26.5), Vec3::Y),
        ..default()
    });
}

fn follow_action(
    carrier: Res<CarrierRes>,
    cfg: Res<CameraConfig>,
    mut cam_query: Query<&mut Transform, With<Camera3d>>,
) {
    let pelvis = nalgebra_vec3_to_bevy(
        carrier.player.skeleton.bodies[SegmentId::Pelvis.idx()].position
    );
    let chest  = nalgebra_vec3_to_bevy(
        carrier.player.skeleton.bodies[SegmentId::Chest.idx()].position
    );
    // Focus on mid-torso (halfway between pelvis and chest) so the camera
    // looks at the part of the body where breathing is visible.
    let focus  = (pelvis + chest) * 0.5;
    let target = pelvis + cfg.offset;

    for mut transform in &mut cam_query {
        let new_pos = spring_follow(transform.translation, target, cfg.alpha);
        *transform = Transform::from_translation(new_pos)
            .looking_at(focus, Vec3::Y);
    }
}
