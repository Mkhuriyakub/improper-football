use physics::math::Vec3;
use physics::xpbd::{XpbdWorld, SOLVER_ITERATIONS};
use physics::rigid_body::RigidBody;
use crate::skeleton::PlayerSkeleton;
use crate::segment::{BodyType, SegmentId};
use crate::locomotion::{StrideGenerator, UpperBodyPose, Role, LegTargets};
use nalgebra::UnitQuaternion;

pub struct Player {
    pub skeleton: PlayerSkeleton,
    pub desired_velocity: Vec3,
    pub role: Role,
    pub elapsed_time: f64,
    pub stride_gen: StrideGenerator,
}

impl Player {
    pub fn new(body_type: BodyType, root: Vec3) -> Self {
        Self {
            skeleton: PlayerSkeleton::new(body_type, root),
            desired_velocity: Vec3::zeros(),
            role: Role::Neutral,
            elapsed_time: 0.0,
            stride_gen: StrideGenerator::default(),
        }
    }

    pub fn step(&mut self, dt: f64) {
        self.elapsed_time += dt;

        let leg_targets = self.stride_gen.compute(self.desired_velocity, self.elapsed_time);
        let pose        = UpperBodyPose::for_role(self.role);

        // Build XPBD world.
        let mut world = XpbdWorld::new();
        for body in std::mem::take(&mut self.skeleton.bodies) { world.add_body(body); }
        world.constraints         = std::mem::take(&mut self.skeleton.constraints);
        world.angular_constraints = std::mem::take(&mut self.skeleton.angular_constraints);

        // ── Substep loop ─────────────────────────────────────────────────────
        // Forces must be re-applied every substep because clear_accumulators is
        // called at the end of each substep. Applying them only once meant 9/10
        // substeps had no gravity cancel and bodies fell under full gravity.
        let sub_dt = dt / SOLVER_ITERATIONS as f64;
        for _ in 0..SOLVER_ITERATIONS {
            // Gravity cancel — every body stays afloat until the ground clamp takes over.
            for body in &mut world.bodies {
                body.apply_force(Vec3::new(0.0, body.mass * 9.81, 0.0));
            }

            // Pelvis height spring (target y = 0.96 m).
            {
                let py  = world.bodies[SegmentId::Pelvis.idx()].position.y;
                let pvy = world.bodies[SegmentId::Pelvis.idx()].linear_velocity.y;
                let f   = (0.96 - py) * 6000.0 - pvy * 400.0;
                world.bodies[SegmentId::Pelvis.idx()]
                    .apply_force(Vec3::new(0.0, f, 0.0));
            }

            // Velocity drive — apply equal acceleration to every body so constraint
            // reactions don't cancel the drive (all bodies accelerate together).
            {
                let pelvis_vel = world.bodies[SegmentId::Pelvis.idx()].linear_velocity;
                let err = Vec3::new(
                    self.desired_velocity.x - pelvis_vel.x,
                    0.0,
                    self.desired_velocity.z - pelvis_vel.z,
                );
                if err.norm() > 0.1 {
                    let raw_accel = err * 40.0;
                    let accel = if raw_accel.norm() > 18.0 {
                        raw_accel.normalize() * 18.0
                    } else {
                        raw_accel
                    };
                    for body in &mut world.bodies {
                        body.apply_force(accel * body.mass);
                    }
                }
            }

            // World-space orientation springs (applied per-substep for stability).
            // Per-substep application keeps equilibrium angular velocity bounded:
            //   ω_eq = (kp/kd) × error, which remains well below MAX_OMEGA at these gains.
            Self::apply_orientation_springs(&mut world.bodies, &leg_targets, &pose);

            world.substep(sub_dt);
        }

        // Recover bodies and constraints.
        self.skeleton.bodies              = world.bodies;
        self.skeleton.constraints         = world.constraints;
        self.skeleton.angular_constraints = world.angular_constraints;

        // ── Angular velocity clamp ──────────────────────────────────────────
        const MAX_OMEGA: f64 = 20.0;
        for body in &mut self.skeleton.bodies {
            let mag = body.angular_velocity.norm();
            if mag > MAX_OMEGA { body.angular_velocity *= MAX_OMEGA / mag; }
        }

        // ── Ground clamp ─────────────────────────────────────────────────────
        let clamps = [
            (SegmentId::FootL,     0.00_f64),
            (SegmentId::FootR,     0.00),
            (SegmentId::LowerLegL, 0.05),
            (SegmentId::LowerLegR, 0.05),
            (SegmentId::UpperLegL, 0.07),
            (SegmentId::UpperLegR, 0.07),
            (SegmentId::Pelvis,    0.12),
        ];
        for (seg, min_y) in clamps {
            let body = &mut self.skeleton.bodies[seg.idx()];
            if body.position.y < min_y {
                body.position.y = min_y;
                if body.linear_velocity.y < 0.0 { body.linear_velocity.y = 0.0; }
            }
        }
    }

    /// World-space orientation spring: Torque = kp×(target×q⁻¹).scaled_axis() − kd×ω
    fn orient_spring(body: &mut RigidBody, target: UnitQuaternion<f64>, kp: f64, kd: f64) {
        let err    = target * body.orientation.inverse();
        let torque = err.scaled_axis() * kp - body.angular_velocity * kd;
        body.apply_torque(torque);
    }

    /// Apply per-substep world-space PD orientation springs to all segments.
    /// Gains are tuned so ω_eq = kp/kd × max_error ≪ MAX_OMEGA.
    fn apply_orientation_springs(
        bodies:  &mut Vec<RigidBody>,
        targets: &LegTargets,
        pose:    &UpperBodyPose,
    ) {
        let upright  = UnitQuaternion::identity();
        let fwd_lean = UnitQuaternion::from_scaled_axis(
            Vec3::new(0.0, 0.0, -pose.torso_lean_forward * 0.4),
        );

        // ── Spine ─────────────────────────────────────────────────────────────
        // kp/kd = 12 → ω_eq ≤ 12 × 0.4 = 4.8 rad/s at max lean
        let spine_kp = 1200.0_f64;
        let spine_kd =  100.0_f64;
        for seg in [SegmentId::Pelvis, SegmentId::Lumbar, SegmentId::Chest,
                    SegmentId::Neck,   SegmentId::Head]
        {
            Self::orient_spring(&mut bodies[seg.idx()], fwd_lean, spine_kp, spine_kd);
        }
        // Extra anti-roll on pelvis.
        Self::orient_spring(&mut bodies[SegmentId::Pelvis.idx()], upright, 400.0, 60.0);

        // ── Arms ──────────────────────────────────────────────────────────────
        let arm_kp  = 400.0_f64;
        let arm_kd  =  60.0_f64;
        let arm_l   = UnitQuaternion::from_scaled_axis(
            Vec3::new(0.0, 0.0,  pose.left_shoulder_flex  * 0.3),
        );
        let arm_r   = UnitQuaternion::from_scaled_axis(
            Vec3::new(0.0, 0.0, -pose.right_shoulder_flex * 0.3),
        );
        Self::orient_spring(&mut bodies[SegmentId::UpperArmL.idx()], arm_l, arm_kp, arm_kd);
        Self::orient_spring(&mut bodies[SegmentId::UpperArmR.idx()], arm_r, arm_kp, arm_kd);
        for seg in [SegmentId::LowerArmL, SegmentId::HandL,
                    SegmentId::LowerArmR, SegmentId::HandR]
        {
            Self::orient_spring(&mut bodies[seg.idx()], upright, arm_kp * 0.5, arm_kd * 0.5);
        }

        // ── Legs ──────────────────────────────────────────────────────────────
        // kp/kd = 12.5 → ω_eq ≤ 12.5 × 1.2 = 15 rad/s at max knee flex (within 20 clamp)
        let hip_l  = targets.hip_flex_l .clamp(-0.52, 0.70);
        let hip_r  = targets.hip_flex_r .clamp(-0.52, 0.70);
        let knee_l = targets.knee_flex_l.clamp( 0.00, 1.20);
        let knee_r = targets.knee_flex_r.clamp( 0.00, 1.20);
        let ank_l  = targets.ankle_flex_l.clamp(-0.30, 0.30);
        let ank_r  = targets.ankle_flex_r.clamp(-0.30, 0.30);

        let leg_kp = 1000.0_f64;
        let leg_kd =   80.0_f64;

        let ul_l = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, -hip_l));
        let ul_r = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, -hip_r));
        Self::orient_spring(&mut bodies[SegmentId::UpperLegL.idx()], ul_l, leg_kp, leg_kd);
        Self::orient_spring(&mut bodies[SegmentId::UpperLegR.idx()], ul_r, leg_kp, leg_kd);

        let ll_l = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, -hip_l + knee_l));
        let ll_r = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, -hip_r + knee_r));
        Self::orient_spring(&mut bodies[SegmentId::LowerLegL.idx()], ll_l, leg_kp, leg_kd);
        Self::orient_spring(&mut bodies[SegmentId::LowerLegR.idx()], ll_r, leg_kp, leg_kd);

        let ft_l = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, -hip_l + knee_l + ank_l));
        let ft_r = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, -hip_r + knee_r + ank_r));
        Self::orient_spring(&mut bodies[SegmentId::FootL.idx()], ft_l, leg_kp * 0.5, leg_kd * 0.5);
        Self::orient_spring(&mut bodies[SegmentId::FootR.idx()], ft_r, leg_kp * 0.5, leg_kd * 0.5);
    }
}
