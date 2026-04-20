use bevy::prelude::*;
use crate::plugins::sim::BallRes;
use crate::render_util::nalgebra_vec3_to_bevy;

#[derive(Component)]
pub struct BallMarker;

pub struct BallPlugin;

impl Plugin for BallPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_ball)
           .add_systems(Update, sync_ball_transform);
    }
}

fn spawn_ball(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = meshes.add(Sphere::new(0.11).mesh().uv(16, 8));
    let mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.6, 0.35, 0.1),
        ..default()
    });
    commands.spawn((
        PbrBundle { mesh, material: mat, ..default() },
        BallMarker,
    ));
}

fn sync_ball_transform(
    ball: Res<BallRes>,
    mut query: Query<&mut Transform, With<BallMarker>>,
) {
    for mut transform in &mut query {
        transform.translation = nalgebra_vec3_to_bevy(ball.ball.body.position);
    }
}
