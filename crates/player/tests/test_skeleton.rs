use player::skeleton::PlayerSkeleton;
use player::segment::{BodyType, SegmentId, SEGMENT_COUNT};
use physics::math::Vec3;

#[test]
fn skeleton_has_correct_segment_count() {
    let skel = PlayerSkeleton::new(BodyType::Receiver, Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(skel.bodies.len(), SEGMENT_COUNT);
}

#[test]
fn head_is_above_feet() {
    let skel = PlayerSkeleton::new(BodyType::Receiver, Vec3::new(0.0, 0.0, 0.0));
    let head_y = skel.bodies[SegmentId::Head.idx()].position.y;
    let foot_y = skel.bodies[SegmentId::FootL.idx()].position.y;
    assert!(head_y > foot_y + 1.4, "head_y={} foot_y={}", head_y, foot_y);
}

#[test]
fn skeleton_offset_by_root_position() {
    let root = Vec3::new(5.0, 0.0, 3.0);
    let skel = PlayerSkeleton::new(BodyType::Lineman, root);
    let pelvis = skel.bodies[SegmentId::Pelvis.idx()].position;
    assert!((pelvis.x - 5.0).abs() < 0.01);
    assert!((pelvis.z - 3.0).abs() < 0.01);
}

#[test]
fn all_bodies_have_positive_mass() {
    let skel = PlayerSkeleton::new(BodyType::Quarterback, Vec3::zeros());
    for (i, body) in skel.bodies.iter().enumerate() {
        assert!(body.mass > 0.0, "body {} has zero mass", i);
    }
}
