use bevy::prelude::*;
use player::segment::{BodyType, SegmentId, segment_def};
use crate::plugins::sim::{CarrierRes, DefenderRes};
use crate::render_util::rigid_body_to_transform;

pub fn segment_capsule_dims(body_type: BodyType, seg: SegmentId) -> (f32, f32) {
    let def = segment_def(body_type, seg);
    (def.radius as f32, def.half_height as f32)
}

#[derive(Component)]
pub struct SegmentMarker {
    pub player_id: u8,
    pub segment_idx: usize,
}

pub struct PlayerPlugin;

impl Plugin for PlayerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_players)
           .add_systems(Update, sync_player_transforms);
    }
}

fn spawn_players(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let teal = materials.add(StandardMaterial {
        base_color: Color::srgb(0.0, 0.7, 0.7),
        ..default()
    });
    let red = materials.add(StandardMaterial {
        base_color: Color::srgb(0.8, 0.1, 0.1),
        ..default()
    });

    let configs: [(u8, BodyType, Handle<StandardMaterial>); 2] = [
        (0u8, BodyType::Receiver, teal),
        (1u8, BodyType::Lineman, red),
    ];

    for (player_id, body_type, mat) in configs {
        for seg in SegmentId::all() {
            let (radius, half_height) = segment_capsule_dims(body_type, seg);
            let mesh = meshes.add(Capsule3d::new(radius, half_height * 2.0));
            commands.spawn((
                PbrBundle {
                    mesh,
                    material: mat.clone(),
                    ..default()
                },
                SegmentMarker { player_id, segment_idx: seg.idx() },
            ));
        }
    }
}

fn sync_player_transforms(
    carrier: Res<CarrierRes>,
    defender: Res<DefenderRes>,
    mut query: Query<(&SegmentMarker, &mut Transform)>,
) {
    for (marker, mut transform) in &mut query {
        let bodies = match marker.player_id {
            0 => &carrier.player.skeleton.bodies,
            _ => &defender.player.skeleton.bodies,
        };
        if let Some(body) = bodies.get(marker.segment_idx) {
            *transform = rigid_body_to_transform(body);
        }
    }
}
