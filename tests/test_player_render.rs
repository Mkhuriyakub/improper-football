use improper_football::plugins::player_render::segment_capsule_dims;
use player::segment::{BodyType, SegmentId};

#[test]
fn all_17_segments_have_valid_dims() {
    for seg in SegmentId::all() {
        let (radius, half_height) = segment_capsule_dims(BodyType::Receiver, seg);
        assert!(radius > 0.0, "{:?} radius must be positive", seg);
        assert!(half_height > 0.0, "{:?} half_height must be positive", seg);
    }
}

#[test]
fn lineman_segments_are_larger_than_receiver() {
    let (r_rec, _) = segment_capsule_dims(BodyType::Receiver, SegmentId::Pelvis);
    let (r_lin, _) = segment_capsule_dims(BodyType::Lineman, SegmentId::Pelvis);
    assert!(r_lin >= r_rec, "lineman pelvis radius {} should be >= receiver {}", r_lin, r_rec);
}
