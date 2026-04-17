use player::player::Player;
use player::segment::{BodyType, SegmentId};
use player::locomotion::Role;
use physics::math::Vec3;

#[test]
fn player_does_not_fall_through_floor_when_standing() {
    let mut player = Player::new(BodyType::Receiver, Vec3::new(0.0, 0.0, 0.0));
    for _ in 0..60 {
        player.step(1.0 / 60.0);
        for seg in [SegmentId::FootL, SegmentId::FootR] {
            if player.skeleton.bodies[seg.idx()].position.y < 0.0 {
                player.skeleton.bodies[seg.idx()].position.y = 0.0;
                player.skeleton.bodies[seg.idx()].linear_velocity.y =
                    player.skeleton.bodies[seg.idx()].linear_velocity.y.max(0.0);
            }
        }
    }
    let pelvis_y = player.skeleton.bodies[SegmentId::Pelvis.idx()].position.y;
    assert!(pelvis_y > 0.3, "pelvis should stay up, got y={}", pelvis_y);
}

#[test]
fn player_moves_in_desired_direction() {
    let mut player = Player::new(BodyType::Receiver, Vec3::zeros());
    player.desired_velocity = Vec3::new(5.0, 0.0, 0.0);
    player.role = Role::WideReceiverRoute;
    for _ in 0..120 {
        player.step(1.0 / 60.0);
        for seg in [SegmentId::FootL, SegmentId::FootR] {
            if player.skeleton.bodies[seg.idx()].position.y < 0.0 {
                player.skeleton.bodies[seg.idx()].position.y = 0.0;
                player.skeleton.bodies[seg.idx()].linear_velocity.y =
                    player.skeleton.bodies[seg.idx()].linear_velocity.y.max(0.0);
            }
        }
    }
    let pelvis_x = player.skeleton.bodies[SegmentId::Pelvis.idx()].position.x;
    assert!(pelvis_x > 0.5, "player should have moved right, got x={}", pelvis_x);
}
