use std::f64::consts::PI;
use physics::math::Vec3;

#[derive(Debug, Clone)]
pub struct LegTargets {
    pub stride_freq: f64,
    pub hip_flex_l: f64,
    pub hip_flex_r: f64,
    pub knee_flex_l: f64,
    pub knee_flex_r: f64,
    pub ankle_flex_l: f64,
    pub ankle_flex_r: f64,
    /// Arm swing angle (negative Z = forward). Arms counter-swing opposite legs.
    pub arm_swing_l: f64,
    pub arm_swing_r: f64,
}

pub struct StrideGenerator {
    pub stride_length: f64,
    pub hip_amplitude: f64,
    pub knee_amplitude: f64,
    pub ankle_amplitude: f64,
}

impl Default for StrideGenerator {
    fn default() -> Self {
        // Phase 1 baseline: zero amplitude — body stands still so we can verify
        // that the physics are stable before adding any movement.
        Self {
            stride_length:  1.4,
            hip_amplitude:  0.00,
            knee_amplitude: 0.00,
            ankle_amplitude: 0.00,
        }
    }
}

impl StrideGenerator {
    pub fn compute(&self, desired_vel: Vec3, t: f64) -> LegTargets {
        let speed = (desired_vel.x.powi(2) + desired_vel.z.powi(2)).sqrt();

        let stride_freq = if speed < 0.1 {
            0.0
        } else {
            (speed / self.stride_length).clamp(0.5, 2.5)
        };

        let phase_l = 2.0 * PI * stride_freq * t;
        let phase_r = phase_l + PI;

        let hip_flex_l   =  self.hip_amplitude * phase_l.sin();
        let hip_flex_r   =  self.hip_amplitude * phase_r.sin();
        // Knee bends during swing (sin > 0 = leg moving forward), straight during stance.
        let knee_flex_l  = self.knee_amplitude * phase_l.sin().max(0.0);
        let knee_flex_r  = self.knee_amplitude * phase_r.sin().max(0.0);
        let ankle_flex_l = self.ankle_amplitude * phase_l.cos();
        let ankle_flex_r = self.ankle_amplitude * phase_r.cos();

        // Arms counter-swing: left arm forward when right leg forward.
        // Amplitude matches typical running arm carriage (~17°).
        let arm_amp     = 0.30_f64;
        let arm_swing_l = -arm_amp * phase_r.sin();
        let arm_swing_r = -arm_amp * phase_l.sin();

        LegTargets { stride_freq, hip_flex_l, hip_flex_r,
                     knee_flex_l, knee_flex_r, ankle_flex_l, ankle_flex_r,
                     arm_swing_l, arm_swing_r }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Role {
    QuarterbackDropback,
    QuarterbackScramble,
    WideReceiverRoute,
    RunningBack,
    Blocker,
    DefenderCoverage,
    DefenderRush,
    Neutral,
}

#[derive(Debug, Clone)]
pub struct UpperBodyPose {
    pub torso_lean_forward: f64,
    pub right_shoulder_flex: f64,
    pub right_elbow_flex: f64,
    pub left_shoulder_flex: f64,
    pub left_elbow_flex: f64,
}

impl UpperBodyPose {
    pub fn for_role(role: Role) -> Self {
        match role {
            Role::QuarterbackDropback => Self {
                torso_lean_forward: 0.15,
                right_shoulder_flex: 0.80,
                right_elbow_flex: 1.30,
                left_shoulder_flex: 0.30,
                left_elbow_flex: 0.80,
            },
            Role::QuarterbackScramble => Self {
                torso_lean_forward: 0.30,
                right_shoulder_flex: 0.20,
                right_elbow_flex: 0.60,
                left_shoulder_flex: 0.20,
                left_elbow_flex: 0.60,
            },
            Role::Blocker => Self {
                torso_lean_forward: 0.40,
                right_shoulder_flex: 0.50,
                right_elbow_flex: 1.20,
                left_shoulder_flex: 0.50,
                left_elbow_flex: 1.20,
            },
            Role::WideReceiverRoute | Role::RunningBack
            | Role::DefenderCoverage | Role::DefenderRush
            | Role::Neutral => Self {
                torso_lean_forward: 0.20,
                right_shoulder_flex: 0.30,
                right_elbow_flex: 0.90,
                left_shoulder_flex: 0.30,
                left_elbow_flex: 0.90,
            },
        }
    }
}
