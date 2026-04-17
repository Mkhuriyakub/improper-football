use physics::math::Vec3;
use physics::xpbd::XpbdWorld;
use crate::skeleton::PlayerSkeleton;
use crate::segment::{BodyType, SegmentId};
use crate::balance::BalanceController;
use crate::locomotion::{StrideGenerator, UpperBodyPose, Role, LegTargets};
use crate::pd_controller::{default_pd, JointKind};

pub struct Player {
    pub skeleton: PlayerSkeleton,
    pub desired_velocity: Vec3,
    pub role: Role,
    pub elapsed_time: f64,
    balance: BalanceController,
    stride_gen: StrideGenerator,
}

impl Player {
    pub fn new(body_type: BodyType, root: Vec3) -> Self {
        Self {
            skeleton: PlayerSkeleton::new(body_type, root),
            desired_velocity: Vec3::zeros(),
            role: Role::Neutral,
            elapsed_time: 0.0,
            balance: BalanceController::default(),
            stride_gen: StrideGenerator::default(),
        }
    }

    pub fn step(&mut self, dt: f64) {
        self.elapsed_time += dt;

        let leg_targets = self.stride_gen.compute(self.desired_velocity, self.elapsed_time);
        let pose = UpperBodyPose::for_role(self.role);

        self.apply_leg_torques(&leg_targets);
        self.apply_upper_body_torques(&pose);
        self.balance.apply_corrections(&mut self.skeleton, dt);

        // Ground support: cancel gravity for all bodies when any foot is on or near the ground.
        // Distributing by mass gives each body zero net vertical acceleration, keeping the
        // skeleton upright. When feet leave the ground (e.g. tackle), gravity resumes.
        let any_foot_grounded =
            self.skeleton.bodies[SegmentId::FootL.idx()].position.y < 0.4 ||
            self.skeleton.bodies[SegmentId::FootR.idx()].position.y < 0.4;
        if any_foot_grounded {
            for body in &mut self.skeleton.bodies {
                let cancel = body.mass * 9.81;
                body.apply_force(Vec3::new(0.0, cancel, 0.0));
            }
        }

        let speed = self.desired_velocity.norm();
        if speed > 0.1 {
            let drive_force = self.desired_velocity.normalize() * 800.0;
            self.skeleton.bodies[SegmentId::Pelvis.idx()].apply_force(drive_force);
        }

        let mut world = XpbdWorld::new();
        let bodies = std::mem::take(&mut self.skeleton.bodies);
        for body in bodies {
            world.add_body(body);
        }
        world.constraints = std::mem::take(&mut self.skeleton.constraints);

        world.step(dt);

        self.skeleton.bodies = world.bodies;
        self.skeleton.constraints = world.constraints;

        // Clamp angular velocities to prevent numerical explosion from high PD gains.
        const MAX_OMEGA: f64 = 30.0;
        for body in &mut self.skeleton.bodies {
            let mag = body.angular_velocity.norm();
            if mag > MAX_OMEGA {
                body.angular_velocity *= MAX_OMEGA / mag;
            }
        }

        // Enforce ground plane on feet: prevent penetration below y=0.
        for seg in [SegmentId::FootL, SegmentId::FootR] {
            let body = &mut self.skeleton.bodies[seg.idx()];
            if body.position.y < 0.0 {
                body.position.y = 0.0;
                if body.linear_velocity.y < 0.0 {
                    body.linear_velocity.y = 0.0;
                }
            }
        }
    }

    fn apply_leg_torques(&mut self, targets: &LegTargets) {
        let pd_hip  = default_pd(JointKind::HipKnee);
        let pd_knee = default_pd(JointKind::HipKnee);
        let pd_ank  = default_pd(JointKind::Ankle);

        let hip_l_vel = self.skeleton.bodies[SegmentId::UpperLegL.idx()].angular_velocity;
        let hip_r_vel = self.skeleton.bodies[SegmentId::UpperLegR.idx()].angular_velocity;
        let t_hip_l = pd_hip.compute_torque(Vec3::new(0.0, 0.0, targets.hip_flex_l), hip_l_vel, Vec3::zeros());
        let t_hip_r = pd_hip.compute_torque(Vec3::new(0.0, 0.0, targets.hip_flex_r), hip_r_vel, Vec3::zeros());
        self.skeleton.bodies[SegmentId::UpperLegL.idx()].apply_torque(t_hip_l);
        self.skeleton.bodies[SegmentId::UpperLegR.idx()].apply_torque(t_hip_r);

        let knee_l_vel = self.skeleton.bodies[SegmentId::LowerLegL.idx()].angular_velocity;
        let knee_r_vel = self.skeleton.bodies[SegmentId::LowerLegR.idx()].angular_velocity;
        let t_knee_l = pd_knee.compute_torque(Vec3::new(0.0, 0.0, targets.knee_flex_l), knee_l_vel, Vec3::zeros());
        let t_knee_r = pd_knee.compute_torque(Vec3::new(0.0, 0.0, targets.knee_flex_r), knee_r_vel, Vec3::zeros());
        self.skeleton.bodies[SegmentId::LowerLegL.idx()].apply_torque(t_knee_l);
        self.skeleton.bodies[SegmentId::LowerLegR.idx()].apply_torque(t_knee_r);

        let ank_l_vel = self.skeleton.bodies[SegmentId::FootL.idx()].angular_velocity;
        let ank_r_vel = self.skeleton.bodies[SegmentId::FootR.idx()].angular_velocity;
        let t_ank_l = pd_ank.compute_torque(Vec3::new(0.0, 0.0, targets.ankle_flex_l), ank_l_vel, Vec3::zeros());
        let t_ank_r = pd_ank.compute_torque(Vec3::new(0.0, 0.0, targets.ankle_flex_r), ank_r_vel, Vec3::zeros());
        self.skeleton.bodies[SegmentId::FootL.idx()].apply_torque(t_ank_l);
        self.skeleton.bodies[SegmentId::FootR.idx()].apply_torque(t_ank_r);
    }

    fn apply_upper_body_torques(&mut self, pose: &UpperBodyPose) {
        let pd_spine = default_pd(JointKind::Spine);
        let chest_vel = self.skeleton.bodies[SegmentId::Chest.idx()].angular_velocity;
        let t_lean = pd_spine.compute_torque(Vec3::new(0.0, 0.0, -pose.torso_lean_forward), chest_vel, Vec3::zeros());
        self.skeleton.bodies[SegmentId::Chest.idx()].apply_torque(t_lean);

        let pd_sh = default_pd(JointKind::Shoulder);
        let r_sh_vel = self.skeleton.bodies[SegmentId::UpperArmR.idx()].angular_velocity;
        let t_r_sh = pd_sh.compute_torque(Vec3::new(0.0, 0.0, pose.right_shoulder_flex), r_sh_vel, Vec3::zeros());
        self.skeleton.bodies[SegmentId::UpperArmR.idx()].apply_torque(t_r_sh);

        let l_sh_vel = self.skeleton.bodies[SegmentId::UpperArmL.idx()].angular_velocity;
        let t_l_sh = pd_sh.compute_torque(Vec3::new(0.0, 0.0, -pose.left_shoulder_flex), l_sh_vel, Vec3::zeros());
        self.skeleton.bodies[SegmentId::UpperArmL.idx()].apply_torque(t_l_sh);
    }
}
