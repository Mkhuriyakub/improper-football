use player::pd_controller::PdController;
use physics::math::Vec3;

#[test]
fn zero_error_produces_zero_torque_at_rest() {
    let pd = PdController { kp: 500.0, kd: 50.0 };
    let torque = pd.compute_torque(Vec3::zeros(), Vec3::zeros(), Vec3::zeros());
    assert!(torque.norm() < 1e-10);
}

#[test]
fn positive_error_produces_positive_torque() {
    let pd = PdController { kp: 500.0, kd: 50.0 };
    let error = Vec3::new(0.1, 0.0, 0.0);
    let torque = pd.compute_torque(error, Vec3::zeros(), Vec3::zeros());
    assert!(torque.x > 0.0, "expected positive torque, got {}", torque.x);
}

#[test]
fn damping_opposes_velocity() {
    let pd = PdController { kp: 0.0, kd: 50.0 };
    let rel_vel = Vec3::new(1.0, 0.0, 0.0);
    let torque = pd.compute_torque(Vec3::zeros(), rel_vel, Vec3::zeros());
    assert!(torque.x < 0.0, "damping should oppose velocity, got {}", torque.x);
}

#[test]
fn torque_scales_with_kp() {
    let pd1 = PdController { kp: 100.0, kd: 0.0 };
    let pd2 = PdController { kp: 200.0, kd: 0.0 };
    let error = Vec3::new(0.5, 0.0, 0.0);
    let t1 = pd1.compute_torque(error, Vec3::zeros(), Vec3::zeros());
    let t2 = pd2.compute_torque(error, Vec3::zeros(), Vec3::zeros());
    assert!((t2.x / t1.x - 2.0).abs() < 1e-6);
}
