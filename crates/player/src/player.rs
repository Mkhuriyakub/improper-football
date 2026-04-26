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
    /// When set, overrides stride generator output for independent appendage testing.
    pub pose_override: Option<LegTargets>,
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
            // 0.96 matches the skeleton's natural init height — zero initial spring
            // error means no transient force to perturb the standing pose.
            pelvis_height_target: 0.96,
            gravity_cancel_active: true,
            pose_override: None,
        }
    }

    pub fn step(&mut self, dt: f64) {
        self.elapsed_time += dt;

        let leg_targets = self.pose_override.clone().unwrap_or_else(|| {
            self.stride_gen.compute(self.desired_velocity, self.elapsed_time)
        });
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

            // Orientation springs only during upright play. Post-tackle the player is a
            // ragdoll — joints fall under gravity + angular limits. Running springs during
            // the fall fight the contact impulses and cause chaotic barrel rolls.
            if self.gravity_cancel_active {
                Self::apply_orientation_springs(&mut world.bodies, &leg_targets, &pose);
            }

            // Remove axial (long-axis) spin from arm/hand bodies every substep.
            // Arms use world-space springs (not relative joint torques) because arm
            // inertia ~0.005 kg·m² makes any reaction torque immediately hit MAX_OMEGA.
            // The positional elbow constraint transmits forearm angular velocity to the
            // upper arm around its local Y axis (the axial direction). Zeroing this
            // component each substep suppresses the accumulation without affecting
            // sagittal swing (around world Z), which is what we actually want to control.
            for seg in [SegmentId::UpperArmL, SegmentId::UpperArmR,
                        SegmentId::LowerArmL, SegmentId::LowerArmR,
                        SegmentId::HandL,     SegmentId::HandR] {
                let body = &mut world.bodies[seg.idx()];
                let arm_long = body.orientation * Vec3::y();
                let axial    = body.angular_velocity.dot(&arm_long);
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

    /// World-space orientation spring for spine/balance: applies torque to one body only.
    /// Represents the postural reflex — intentionally not Newton's-3rd-law balanced
    /// because the vestibular ground reaction is the implicit counter-force.
    fn orient_spring(body: &mut RigidBody, target: UnitQuaternion<f64>, kp: f64, kd: f64) {
        let err    = target * body.orientation.inverse();
        let torque = err.scaled_axis() * kp - body.angular_velocity * kd;
        body.apply_torque(torque);
    }

    /// Relative joint torque spring (Newton's 3rd law muscle model).
    /// +torque applied to child, −torque applied to parent.
    /// Net angular momentum added to the whole skeleton = 0, so leg/arm spring torques
    /// cannot propagate through the constraint chain and spin the trunk.
    fn joint_spring_rel(
        bodies:     &mut Vec<RigidBody>,
        parent: usize, child: usize,
        target_rel: UnitQuaternion<f64>,
        kp: f64, kd: f64,
    ) {
        // Copy out what we need before taking any mut borrows.
        let q_p     = bodies[parent].orientation;
        let q_c     = bodies[child].orientation;
        let omega_p = bodies[parent].angular_velocity;
        let omega_c = bodies[child].angular_velocity;

        // Actual relative rotation (child in parent's frame) and error vs target.
        let q_rel        = q_p.inverse() * q_c;
        let err          = (target_rel * q_rel.inverse()).scaled_axis();
        let omega_rel    = omega_c - omega_p;
        let torque_world = q_p * (err * kp - omega_rel * kd);

        bodies[child].apply_torque( torque_world);
        bodies[parent].apply_torque(-torque_world);
    }

    /// Apply per-substep orientation springs to all segments.
    ///
    /// Spine  — world-space balance springs (postural reflex; one body at a time).
    /// Legs & arms — relative joint torques (Newton's 3rd law) so that gait
    ///   spring torques are internally balanced and cannot pump angular momentum
    ///   into the trunk via positional constraint reactions.
    fn apply_orientation_springs(
        bodies:  &mut Vec<RigidBody>,
        targets: &LegTargets,
        pose:    &UpperBodyPose,
    ) {
        let upright = UnitQuaternion::identity();

        // Lean angle scales with stride frequency so the body stands fully upright
        // at rest (zero initial spring error) and leans forward when running.
        let lean_scale = (targets.stride_freq / 2.5).clamp(0.0, 1.0);
        let spine_target = UnitQuaternion::from_scaled_axis(
            Vec3::new(0.0, 0.0, -pose.torso_lean_forward * 0.4 * lean_scale),
        );

        // ── Spine: world-space balance ────────────────────────────────────────
        let spine_kp = 1200.0_f64;
        let spine_kd =  150.0_f64;
        for seg in [SegmentId::Pelvis, SegmentId::Lumbar, SegmentId::Chest,
                    SegmentId::Neck,   SegmentId::Head]
        {
            Self::orient_spring(&mut bodies[seg.idx()], spine_target, spine_kp, spine_kd);
        }
        // Extra upright spring on pelvis only — resist lateral/twist drift.
        Self::orient_spring(&mut bodies[SegmentId::Pelvis.idx()], upright, 500.0, 80.0);

        // ── Arms: world-space springs (arm inertia ≈ 0.005 kg·m² is too small for
        //   relative joint springs — the reaction torque on UpperArm from an elbow
        //   spring hits MAX_OMEGA within one substep at any useful gain).
        //   Axial (long-axis) spin is removed each substep in the outer loop instead.
        let arm_kp     =  50.0_f64;
        let arm_kd     =  60.0_f64;  // increased: prevents bouncing at large arm angles
        // Elbow bend scales with stride frequency: arms hang straight at rest,
        // flex to ~40° during running (same lean_scale used for spine).
        let elbow_bend = 0.70_f64 * lean_scale;

        let arm_l = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, targets.arm_swing_l));
        let arm_r = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, targets.arm_swing_r));
        Self::orient_spring(&mut bodies[SegmentId::UpperArmL.idx()], arm_l, arm_kp, arm_kd);
        Self::orient_spring(&mut bodies[SegmentId::UpperArmR.idx()], arm_r, arm_kp, arm_kd);

        let la_l = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, targets.arm_swing_l + elbow_bend));
        let la_r = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, targets.arm_swing_r + elbow_bend));
        Self::orient_spring(&mut bodies[SegmentId::LowerArmL.idx()], la_l, arm_kp, arm_kd);
        Self::orient_spring(&mut bodies[SegmentId::LowerArmR.idx()], la_r, arm_kp, arm_kd);
        Self::orient_spring(&mut bodies[SegmentId::HandL.idx()], upright, arm_kp * 0.4, arm_kd * 0.4);
        Self::orient_spring(&mut bodies[SegmentId::HandR.idx()], upright, arm_kp * 0.4, arm_kd * 0.4);

        // ── Legs: relative joint torques (anatomical angles) ─────────────────
        // hip_flex > 0  → leg forward  (−Z rotation of UpperLeg relative to Pelvis)
        // knee_flex > 0 → knee bends   (+Z rotation of LowerLeg relative to UpperLeg)
        // ankle_flex > 0 → dorsiflexion (+Z rotation of Foot relative to LowerLeg)
        let hip_l  = targets.hip_flex_l .clamp(-0.52, 0.70);
        let hip_r  = targets.hip_flex_r .clamp(-0.52, 0.70);
        let knee_l = targets.knee_flex_l.clamp( 0.00, 1.40);
        let knee_r = targets.knee_flex_r.clamp( 0.00, 1.40);
        let ank_l  = targets.ankle_flex_l.clamp(-0.40, 0.40);
        let ank_r  = targets.ankle_flex_r.clamp(-0.40, 0.40);

        let leg_kp  = 600.0_f64;
        let leg_kd  =  80.0_f64;
        let foot_kp = 250.0_f64;
        let foot_kd =  60.0_f64;

        let hip_l_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, -hip_l));
        let hip_r_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, -hip_r));
        Self::joint_spring_rel(bodies, SegmentId::Pelvis.idx(),   SegmentId::UpperLegL.idx(), hip_l_rel,  leg_kp, leg_kd);
        Self::joint_spring_rel(bodies, SegmentId::Pelvis.idx(),   SegmentId::UpperLegR.idx(), hip_r_rel,  leg_kp, leg_kd);

        let knee_l_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, knee_l));
        let knee_r_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, knee_r));
        Self::joint_spring_rel(bodies, SegmentId::UpperLegL.idx(), SegmentId::LowerLegL.idx(), knee_l_rel, leg_kp, leg_kd);
        Self::joint_spring_rel(bodies, SegmentId::UpperLegR.idx(), SegmentId::LowerLegR.idx(), knee_r_rel, leg_kp, leg_kd);

        let ank_l_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, ank_l));
        let ank_r_rel = UnitQuaternion::from_scaled_axis(Vec3::new(0.0, 0.0, ank_r));
        Self::joint_spring_rel(bodies, SegmentId::LowerLegL.idx(), SegmentId::FootL.idx(), ank_l_rel, foot_kp, foot_kd);
        Self::joint_spring_rel(bodies, SegmentId::LowerLegR.idx(), SegmentId::FootR.idx(), ank_r_rel, foot_kp, foot_kd);
    }
}
