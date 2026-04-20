mod render_util;
mod plugins;

use bevy::prelude::*;
use plugins::{
    sim::SimPlugin,
    field::FieldPlugin,
    player_render::PlayerPlugin,
    ball_render::BallPlugin,
    camera::CameraPlugin,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Improper Football".into(),
                resolution: (1280.0, 720.0).into(),
                ..default()
            }),
            ..default()
        }))
        .add_plugins((SimPlugin, FieldPlugin, PlayerPlugin, BallPlugin, CameraPlugin))
        .run();
}
