use improper_football::plugins::sim::{CarrierRes, DefenderRes, SimConfig};

#[test]
fn sim_config_dt_is_120hz() {
    let cfg = SimConfig::default();
    assert!((cfg.dt - 1.0_f64 / 120.0).abs() < 1e-10);
}

#[test]
fn carrier_resource_can_step() {
    let mut res = CarrierRes::default();
    res.player.step(1.0 / 120.0);
}

#[test]
fn defender_resource_can_step() {
    let mut res = DefenderRes::default();
    res.player.step(1.0 / 120.0);
}
