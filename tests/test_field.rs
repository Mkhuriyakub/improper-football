use improper_football::plugins::field::{FIELD_LENGTH, FIELD_WIDTH, yard_line_x_positions};

#[test]
fn field_dimensions_are_correct() {
    assert!((FIELD_LENGTH - 100.0_f32).abs() < 0.01);
    assert!((FIELD_WIDTH - 53.3_f32).abs() < 0.1);
}

#[test]
fn ten_yard_lines_count_is_nine() {
    let positions = yard_line_x_positions();
    assert_eq!(positions.len(), 9);
}

#[test]
fn yard_lines_are_evenly_spaced() {
    let positions = yard_line_x_positions();
    for (i, &x) in positions.iter().enumerate() {
        let expected = (i + 1) as f32 * 10.0;
        assert!((x - expected).abs() < 0.01,
            "yard line {} should be at x={}, got x={}", i, expected, x);
    }
}
