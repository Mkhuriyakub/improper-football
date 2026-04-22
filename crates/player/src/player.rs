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
    /// 1.0 = full drive, 0.0 = disabled when tackled.
    pub velocity_drive_scale: f64,
    /// Pelvis height spring target. Lowered during a tackle to pull the carrier to ground.
    pub pelvis_height_target: f64,
    /// When false, gravity cancel is off and bodies fall under real gravity (tackle state).
    pub gravity_cancel_active: bool,
}

impl Player {
    pub fn new(body_type: BodyType, root: Vec3) -> Self {
        Self {
            skeleton: PlayerSkeleton::new(body_type, root),
            desired_velocity: Vec3::zeros(),
            role: Role::Neutral,
            elapsed_time: 0.0,
            stride_gen: StrideGenerator::default(),
            velocity_drive_scale: 1.0,
            pelvis_height_target: 0.90,
            gravity_cancel_active: true,
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
            // Gravity cancel keeps bodies weightless during normal play.
            // Disabled when tackled so the player falls under real gravity.
            if self.gravity_cancel_active {
                for body in &mut world.bodies {
                    body.apply_force(Vec3::new(0.0, body.mass * 9.81, 0.0));
                }
            }

            // Pelvis height spring. Target drops during tackle to pull the carrier to the ground.
            {
                let py  = world.bodies[SegmentId::Pelvis.idx()].position.y;
                let pvy = world.bodies[SegmentId::Pelvis.idx()].linear_velocity.y;
                let f   = (self.pelvis_height_target - py) * 6000.0 - pvy * 400.0;
                world.bodies[SegmentId::Pelvis.idx()]
                    .apply_force(Vec3::new(0.0, f, 0.0));
            }

            // Velocity drive — apply equal acceleration to every body so constraint
            // reactions don't cancel the drive (all bodies accelerate together).
            // velocity_drive_scale = 0 when tackled, disabling drive so collision forces dominate.
            {
                let pelvis_vel = world.bodies[SegmentId::Pelvis.idx()].linear_velocity;
                let err = Vec3::new(
                    self.desired_velocity.x - pelvis_vel.x,
                    0.0,
                    self.desired_velocity.z - pelvis_vel.z,
                );
                let raw_accel = err * 20.0;
                let accel = (if raw_accel.norm() > 10.0 {
                    raw_accel.normalize() * 10.0
                } else {
                    raw_accel
                }) * self.velocity_drive_scale;
                for body in &mut world.bodies {
                    body.apply_force(accel * body.mass);
                }
            }

            // World-space orientation springs (applied per-substep for stability).
            // Per-substep application keeps equilibrium angular velocity bounded:
            //   ω_eq = (kp/kd) × error, which remains well below MAX_OMEGA at these gains.
            Self::apply_orientation_springs(&mut world.bodies, &leg_targets, &pose);

            // Remove axial spin from arm/hand bodies every substep.
            // The leg springs transmit torque through the constraint chain, spinning arms
            // around their long axis (local Y). I_yy ≈ 0.002 kg·m² so even tiny torques
            // reach MAX_OMEGA instantly. Zeroing the axial component each substep leaves
            // sagittal swing (local Z/X) untouched.
            for seg in [SegmentId::UpperArmL, SegmentId::UpperArmR,
                        SegmentId::LowerArmL, SegmentId::LowerArmR,
                        SegmentId::HandL,     SegmentId::HandR] {
                let body = &mut world.bodies[seg.idx()];
                let arm_long = body.orientation * Vec3::y();
                let axial = body.angular_velocity.dot(&arm_long);
                body.angular_velocity -= arm_long * axial;
            }

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
            (SegmentId::Pelvis,    0.10),
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
        let spine_kp = 1200.0_f64;
        let spine_kd =  100.0_f64;
        for seg in [SegmentId::Pelvis, SegmentId::Lumbar, SegmentId::Chest,
                    SegmentId::Neck,   SegmentId::Head]
        {
            Self::orient_spring(&mut bodies[seg.idx()], fwd_lean, spine_kp, spine_kd);
        }

        Self::orient_spring(&mut bodies[SegmentId::Pelvis.idx()], upright, 400.0, 60.0);

        // ── Arms — counter-swing opposite to legs, elbow gently bent ─────────
        // Arm inertia is tiny (~0.0085 kg·m²), so gains must be very low.
        // kp/kd = 50/30 → ω_eq ≤ 1.7 × 0.22 = 0.37 rad/s — no flailing possible.
        let arm_kp   =  50.0_f64;
        let arm_kd   =  30.0_f64;

        // Upper arm swings in sagittal plane (Z axis).
        let arm_l = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, targets.arm_swing_l));
        let arm_r = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, targets.arm_swing_r));
        Self::orient_spring(&mut bodies[SegmentId::UpperArmL.idx()], arm_l, arm_kp, arm_kd);
        Self::orient_spring(&mut bodies[SegmentId::UpperArmR.idx()], arm_r, arm_kp, arm_kd);

        // Forearm: gentle elbow bend (0.40 rad ≈ 23° of additional Z rotation).
        // Kept small to avoid large initial error — arms start at identity orientation.
        let elbow_bend = 0.40_f64;
        let la_l = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, targets.arm_swing_l + elbow_bend));
        let la_r = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, targets.arm_swing_r + elbow_bend));
        Self::orient_spring(&mut bodies[SegmentId::LowerArmL.idx()], la_l, arm_kp, arm_kd);
        Self::orient_spring(&mut bodies[SegmentId::LowerArmR.idx()], la_r, arm_kp, arm_kd);
        Self::orient_spring(&mut bodies[SegmentId::HandL.idx()], upright, arm_kp * 0.4, arm_kd * 0.4);
        Self::orient_spring(&mut bodies[SegmentId::HandR.idx()], upright, arm_kp * 0.4, arm_kd * 0.4);

        // ── Legs ──────────────────────────────────────────────────────────────
        // kp=550 tracks smoothly without snapping (ω_eq = 550/75 × 0.25 = 1.8 rad/s).
        let hip_l  = targets.hip_flex_l .clamp(-0.52, 0.70);
        let hip_r  = targets.hip_flex_r .clamp(-0.52, 0.70);
        let knee_l = targets.knee_flex_l.clamp( 0.00, 1.20);
        let knee_r = targets.knee_flex_r.clamp( 0.00, 1.20);
        let ank_l  = targets.ankle_flex_l.clamp(-0.30, 0.30);
        let ank_r  = targets.ankle_flex_r.clamp(-0.30, 0.30);

        let leg_kp  = 550.0_f64;
        let leg_kd  =  75.0_f64;
        let foot_kp = 220.0_f64;
        let foot_kd =  55.0_f64;

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
        Self::orient_spring(&mut bodies[SegmentId::FootL.idx()], ft_l, foot_kp, foot_kd);
        Self::orient_spring(&mut bodies[SegmentId::FootR.idx()], ft_r, foot_kp, foot_kd);
    }
}
