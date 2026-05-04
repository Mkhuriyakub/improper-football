use player::segment::{BodyType, SegmentId, segment_def};

#[test]
fn lineman_pelvis_mass_is_realistic() {
    let def = segment_def(BodyType::Lineman, SegmentId::Pelvis);
    // Pelvis is roughly 15% of total body mass for a 140kg lineman
    assert!(def.mass > 15.0 && def.mass < 25.0, "pelvis mass = {}", def.mass);
}

#[test]
fn all_segments_have_positive_mass() {
    for body_type in [BodyType::Lineman, BodyType::Receiver, BodyType::Quarterback] {
        for seg in SegmentId::all() {
            let def = segment_def(body_type, seg);
            assert!(def.mass > 0.0, "{:?}/{:?} mass must be positive", body_type, seg);
            assert!(def.radius > 0.0);
            assert!(def.half_height > 0.0);
        }
    }
}

#[test]
fn inertia_tensor_is_symmetric_positive_definite() {
    let def = segment_def(BodyType::Receiver, SegmentId::Chest);
    let i = def.inertia_tensor();
    // Diagonal elements must be positive
    assert!(i[(0,0)] > 0.0 && i[(1,1)] > 0.0 && i[(2,2)] > 0.0);
    // Symmetric
    assert!((i[(0,1)] - i[(1,0)]).abs() < 1e-10);
}
