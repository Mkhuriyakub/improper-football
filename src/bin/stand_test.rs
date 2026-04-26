use player::player::Player;
use player::segment::{BodyType, SegmentId};
use player::locomotion::{Role, LegTargets};
use physics::math::Vec3;

const DT: f64 = 1.0 / 120.0;
const ARM_SKY:    f64 = -2.50;  // rad — arm raised ~143° from hanging (well past horizontal)
const RAMP_SECS:  f64 = 4.0;    // seconds to ramp from 0 to ARM_SKY

fn make_pose(arm_l: f64, arm_r: f64) -> LegTargets {
    LegTargets {
        stride_freq: 0.0,
        hip_flex_l: 0.0, hip_flex_r: 0.0,
        knee_flex_l: 0.0, knee_flex_r: 0.0,
        ankle_flex_l: 0.0, ankle_flex_r: 0.0,
        arm_swing_l: arm_l,
        arm_swing_r: arm_r,
    }
}

fn ramp(t_since_start: f64) -> f64 {
    ARM_SKY * (t_since_start / RAMP_SECS).clamp(0.0, 1.0)
}

fn main() {
    let mut player = Player::new(BodyType::Receiver, Vec3::new(0.0, 0.0, 0.0));
    player.desired_velocity = Vec3::zeros();
    player.role = Role::RunningBack;

    println!("ARMS-TO-SKY TEST  (target {:.2} rad, ramp {:.0}s)",
             ARM_SKY, RAMP_SECS);
    println!("{:<6} {:<9} {:<8} {:<10} {:<10} {:<10} {:<10}",
             "t", "phase", "target", "uarmL_z", "uarmR_z", "chest_z", "pelv_z");
    println!("{}", "─".repeat(70));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    // Phase 1: 0–2s   neutral
    // Phase 2: 2–6s   left arm ramps to ARM_SKY
    // Phase 3: 6–10s  right arm also ramps to ARM_SKY
    while t < 10.0 {
        let arm_l = if t >= 2.0 { ramp(t - 2.0) } else { 0.0 };
        let arm_r = if t >= 6.0 { ramp(t - 6.0) } else { 0.0 };
        player.pose_override = Some(make_pose(arm_l, arm_r));

        player.step(DT);
        t += DT;

        if t >= next_print {
            let uarm_l = player.skeleton.bodies[SegmentId::UpperArmL.idx()]
                             .orientation.scaled_axis().z;
            let uarm_r = player.skeleton.bodies[SegmentId::UpperArmR.idx()]
                             .orientation.scaled_axis().z;
            let chest_z = player.skeleton.bodies[SegmentId::Chest.idx()]
                              .orientation.scaled_axis().z;
            let pelv_z  = player.skeleton.bodies[SegmentId::Pelvis.idx()]
                              .orientation.scaled_axis().z;

            let phase = if t < 2.0 { "neutral" }
                        else if t < 6.0 { "L-ramp" }
                        else { "both" };
            let tgt = if t < 2.0 { 0.0 }
                      else if t < 6.0 { ramp(t - 2.0) }
                      else { ramp(t - 6.0) }; // show right arm target

            println!("{:<6.2} {:<9} {:<8.3} {:<10.4} {:<10.4} {:<10.4} {:<10.4}",
                     t, phase, tgt, uarm_l, uarm_r, chest_z, pelv_z);
            next_print += 0.5;
        }
    }
}
