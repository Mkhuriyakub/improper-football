use bevy::prelude::*;

pub const FIELD_LENGTH: f32 = 100.0;
pub const FIELD_WIDTH: f32 = 53.3;
const STRIPE_HEIGHT: f32 = 0.02;
const STRIPE_THICKNESS: f32 = 0.15;

pub fn yard_line_x_positions() -> Vec<f32> {
    (1..=9).map(|i| i as f32 * 10.0).collect()
}

pub struct FieldPlugin;

impl Plugin for FieldPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_field);
    }
}

fn spawn_field(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 200.0,
    });

    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 10_000.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(50.0, 80.0, 30.0)
            .looking_at(Vec3::new(50.0, 0.0, 26.0), Vec3::Y),
        ..default()
    });

    let field_mesh = meshes.add(Plane3d::default().mesh().size(FIELD_LENGTH, FIELD_WIDTH));
    let green = materials.add(StandardMaterial {
        base_color: Color::srgb(0.13, 0.55, 0.13),
        ..default()
    });
    commands.spawn(PbrBundle {
        mesh: field_mesh,
        material: green,
        transform: Transform::from_xyz(FIELD_LENGTH * 0.5, 0.0, FIELD_WIDTH * 0.5),
        ..default()
    });

    let end_zone_mat = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.38, 0.08),
        ..default()
    });
    for x_centre in [5.0_f32, 95.0_f32] {
        let ez_mesh = meshes.add(Plane3d::default().mesh().size(10.0, FIELD_WIDTH));
        commands.spawn(PbrBundle {
            mesh: ez_mesh,
            material: end_zone_mat.clone(),
            transform: Transform::from_xyz(x_centre, 0.01, FIELD_WIDTH * 0.5),
            ..default()
        });
    }

    let stripe_mesh = meshes.add(Cuboid::new(STRIPE_THICKNESS, STRIPE_HEIGHT, FIELD_WIDTH));
    let white = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        ..default()
    });
    for x in yard_line_x_positions() {
        commands.spawn(PbrBundle {
            mesh: stripe_mesh.clone(),
            material: white.clone(),
            transform: Transform::from_xyz(x, 0.02, FIELD_WIDTH * 0.5),
            ..default()
        });
    }
}
