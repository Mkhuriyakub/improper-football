use player::player::Player;
use player::segment::{BodyType, SegmentId};
use player::locomotion::Role;
use physics::math::Vec3;

const DT: f64 = 1.0 / 120.0;
const SIM_DURATION: f64 = 3.0;
const PRINT_INTERVAL: f64 = 0.25;

fn clamp_feet_to_ground(player: &mut Player) {
    for seg in [SegmentId::FootL, SegmentId::FootR] {
        let body = &mut player.skeleton.bodies[seg.idx()];
        if body.position.y < 0.0 {
            body.position.y = 0.0;
            if body.linear_velocity.y < 0.0 {
                body.linear_velocity.y = 0.0;
            }
        }
    }
}

fn pelvis_y(player: &Player) -> f64 {
    player.skeleton.bodies[SegmentId::Pelvis.idx()].position.y
}

fn foot_y(player: &Player, seg: SegmentId) -> f64 {
    player.skeleton.bodies[seg.idx()].position.y
}

fn arm_ang(player: &Player, seg: SegmentId) -> f64 {
    player.skeleton.bodies[seg.idx()].angular_velocity.norm()
}

fn pelvis_x(player: &Player) -> f64 {
    player.skeleton.bodies[SegmentId::Pelvis.idx()].position.x
}

fn main() {
    let mut carrier = Player::new(BodyType::Receiver, Vec3::new(0.0, 0.0, 0.0));
    carrier.desired_velocity = Vec3::new(5.0, 0.0, 0.0);
    carrier.role = Role::RunningBack;

    let mut defender = Player::new(BodyType::Lineman, Vec3::new(4.0, 0.0, 0.0));
    defender.desired_velocity = Vec3::zeros();
    defender.role = Role::DefenderRush;

    println!("TACKLE SIMULATION");
    println!("{:<7} {:<8} {:<7} {:<7} {:<8} {:<8} {:<8}",
             "t", "pel_y", "fL", "fR", "GAIT", "armL_ω", "armR_ω");
    println!("{}", "-".repeat(62));

    let mut t = 0.0;
    let mut next_print = 0.0;

    while t < SIM_DURATION {
        let c_pos = carrier.skeleton.bodies[SegmentId::Pelvis.idx()].position;
        let d_pos = defender.skeleton.bodies[SegmentId::Pelvis.idx()].position;
        let dist = (c_pos - d_pos).norm();

        if dist < 0.8 {
            let contact_dir = if dist > 1e-6 { (d_pos - c_pos).normalize() } else { Vec3::x() };
            let impulse_mag = 600.0 * DT;
            carrier.skeleton.bodies[SegmentId::Pelvis.idx()]
                .apply_impulse(-contact_dir * impulse_mag, c_pos);
            defender.skeleton.bodies[SegmentId::Pelvis.idx()]
                .apply_impulse(contact_dir * impulse_mag, d_pos);

            carrier.skeleton.bodies[SegmentId::Chest.idx()]
                .apply_force(Vec3::new(0.0, -2000.0, 0.0));
        }

        carrier.step(DT);
        defender.step(DT);

        clamp_feet_to_ground(&mut carrier);
        clamp_feet_to_ground(&mut defender);

        t += DT;

        if t >= next_print {
            let fl = foot_y(&carrier, SegmentId::FootL);
            let fr = foot_y(&carrier, SegmentId::FootR);
            let gait = match (fl < 0.06, fr < 0.06) {
                (true, false)  => "L-sta",
                (false, true)  => "R-sta",
                (true, true)   => "both-gnd",
                (false, false) => "AIRBORNE",
            };
            println!("{:<7.2} {:<8.3} {:<7.3} {:<7.3} {:<8} {:<8.2} {:<8.2}",
                     t, pelvis_y(&carrier), fl, fr, gait,
                     arm_ang(&carrier, SegmentId::UpperArmL),
                     arm_ang(&carrier, SegmentId::UpperArmR));
            next_print += PRINT_INTERVAL;
        }
    }

    println!("\nFinal carrier pelvis Y: {:.3}m", pelvis_y(&carrier));
    if pelvis_y(&carrier) < 0.5 {
        println!("TACKLE: carrier driven to ground.");
    } else {
        println!("Carrier stayed upright (tune balance/contact params).");
    }
}
