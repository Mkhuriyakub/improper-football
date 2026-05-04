// Diagnostic: bare three-phalanx chain with ONLY elastic joints (no angular
// biomech, no tendon). Tests whether the elastic-joint chain itself is
// stable for finger-scale bodies before layering on biomech.

use physics::xpbd::XpbdWorld;
use physics::rigid_body::RigidBody;
use physics::positional_spring::PositionalSpring;
use physics::positional_damper::PositionalDamper;
use physics::math::{Vec3, Mat3};

const DT: f64 = 1.0 / 120.0;
const SOLVER_ITERATIONS: usize = 60;
const SUB_DT: f64 = DT / SOLVER_ITERATIONS as f64;
const SIM_DURATION: f64 = 1.0;
const PRINT_INTERVAL: f64 = 0.025;

const HAND_REST_POS: Vec3 = Vec3::new(0.0, 1.0, 0.0);
const HAND_HALF_H:  f64 = 0.050;

const PROX_MASS: f64 = 0.008;
const PROX_HALF_H: f64 = 0.022;
const MID_MASS:  f64 = 0.005;
const MID_HALF_H:  f64 = 0.012;
const DIST_MASS: f64 = 0.003;
const DIST_HALF_H: f64 = 0.008;

const K: f64 = 10_000.0;
// Per-pair damping for ζ=0.7 (well-damped, stable):
//   c = 1.4 · √(m_red · k)  where m_red is reduced mass for each pair.
// hand-prox (m_red ≈ 0.008): c ≈ 12.5
// prox-mid  (m_red ≈ 0.00308): c ≈ 7.8
// mid-dist  (m_red ≈ 0.00188): c ≈ 6.1
const C_HAND_PROX: f64 = 12.5;
const C_PROX_MID:  f64 = 7.8;
const C_MID_DIST:  f64 = 6.1;

fn capsule_inertia(mass: f64, radius: f64, half_h: f64) -> Mat3 {
    let h = 2.0 * half_h;
    let i_axial = 0.5 * mass * radius * radius;
    let i_trans = (1.0 / 12.0) * mass * (3.0 * radius * radius + h * h);
    Mat3::new(
        i_trans, 0.0,     0.0,
        0.0,     i_axial, 0.0,
        0.0,     0.0,     i_trans,
    )
}

fn main() {
    let mut world = XpbdWorld::new();

    // Hand pinned (mass = 0).
    let mut hand = RigidBody::new(0.0, capsule_inertia(0.54, 0.030, HAND_HALF_H));
    hand.position = HAND_REST_POS;
    let hand_idx = world.add_body(hand);

    // Phalanges, all in straight-down configuration (no angular flex).
    let prox_y = HAND_REST_POS.y - HAND_HALF_H - PROX_HALF_H;
    let mid_y = prox_y - PROX_HALF_H - MID_HALF_H;
    let dist_y = mid_y - MID_HALF_H - DIST_HALF_H;

    let mut prox = RigidBody::new(PROX_MASS, capsule_inertia(PROX_MASS, 0.010, PROX_HALF_H));
    prox.position = Vec3::new(0.0, prox_y, 0.0);
    let prox_idx = world.add_body(prox);

    let mut mid = RigidBody::new(MID_MASS, capsule_inertia(MID_MASS, 0.009, MID_HALF_H));
    mid.position = Vec3::new(0.0, mid_y, 0.0);
    let mid_idx = world.add_body(mid);

    let mut dist = RigidBody::new(DIST_MASS, capsule_inertia(DIST_MASS, 0.008, DIST_HALF_H));
    dist.position = Vec3::new(0.0, dist_y, 0.0);
    let dist_idx = world.add_body(dist);

    // Three elastic joints, damping scaled per pair.
    let setup_joint = |world: &mut XpbdWorld, a: usize, b: usize, ah: f64, bh: f64, c: f64| {
        let mut s = PositionalSpring::new(a, b, K);
        s.local_anchor_a = Vec3::new(0.0, -ah, 0.0);
        s.local_anchor_b = Vec3::new(0.0,  bh, 0.0);
        world.positional_springs.push(s);
        let mut d = PositionalDamper::new(a, b, c);
        d.local_anchor_a = Vec3::new(0.0, -ah, 0.0);
        d.local_anchor_b = Vec3::new(0.0,  bh, 0.0);
        world.positional_dampers.push(d);
    };
    setup_joint(&mut world, hand_idx, prox_idx, HAND_HALF_H, PROX_HALF_H, C_HAND_PROX);
    setup_joint(&mut world, prox_idx, mid_idx, PROX_HALF_H, MID_HALF_H, C_PROX_MID);
    setup_joint(&mut world, mid_idx, dist_idx, MID_HALF_H, DIST_HALF_H, C_MID_DIST);

    println!("FINGER DIAGNOSTIC: bare elastic chain ({:.1} s, {} substeps)",
             SIM_DURATION, SOLVER_ITERATIONS);
    println!();
    println!("{:<7} {:<11} {:<11} {:<11}",
             "t", "prox_y", "mid_y", "dist_y");
    println!("{}", "-".repeat(45));

    let mut t = 0.0;
    let mut next_print = 0.0_f64;

    while t < SIM_DURATION {
        for _ in 0..SOLVER_ITERATIONS {
            world.substep(SUB_DT);
        }
        t += DT;

        if t >= next_print {
            let prox = &world.bodies[prox_idx];
            let mid = &world.bodies[mid_idx];
            let dist = &world.bodies[dist_idx];
            println!("{:<7.4} {:<11.6} {:<11.6} {:<11.6}",
                     t, prox.position.y, mid.position.y, dist.position.y);
            next_print += PRINT_INTERVAL;
        }
    }
}
