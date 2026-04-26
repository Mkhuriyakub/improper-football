use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
use physics::angular_constraint::AngularLimitConstraint;
use physics::math::Vec3;
use crate::segment::{BodyType, SegmentId, SEGMENT_COUNT, segment_def};

pub const JOINT_COUNT: usize = 16;

struct JointDef {
    parent: SegmentId,
    child: SegmentId,
    anchor_in_parent: Vec3,
    anchor_in_child: Vec3,
}

fn joint_defs() -> [JointDef; JOINT_COUNT] {
    [
        // Spine chain
        JointDef { parent: SegmentId::Pelvis, child: SegmentId::Lumbar,
            anchor_in_parent: Vec3::new(0.0,  0.10, 0.0),
            anchor_in_child:  Vec3::new(0.0, -0.09, 0.0) },
        JointDef { parent: SegmentId::Lumbar, child: SegmentId::Chest,
            anchor_in_parent: Vec3::new(0.0,  0.09, 0.0),
            anchor_in_child:  Vec3::new(0.0, -0.14, 0.0) },
        JointDef { parent: SegmentId::Chest, child: SegmentId::Neck,
            anchor_in_parent: Vec3::new(0.0,  0.14, 0.0),
            anchor_in_child:  Vec3::new(0.0, -0.06, 0.0) },
        JointDef { parent: SegmentId::Neck, child: SegmentId::Head,
            anchor_in_parent: Vec3::new(0.0,  0.06, 0.0),
            anchor_in_child:  Vec3::new(0.0, -0.10, 0.0) },
        // Left arm
        JointDef { parent: SegmentId::Chest, child: SegmentId::UpperArmL,
            anchor_in_parent: Vec3::new(-0.14,  0.10, 0.0),
            anchor_in_child:  Vec3::new(  0.0,  0.15, 0.0) },
        JointDef { parent: SegmentId::UpperArmL, child: SegmentId::LowerArmL,
            anchor_in_parent: Vec3::new(0.0, -0.15, 0.0),
            anchor_in_child:  Vec3::new(0.0,  0.13, 0.0) },
        JointDef { parent: SegmentId::LowerArmL, child: SegmentId::HandL,
            anchor_in_parent: Vec3::new(0.0, -0.13, 0.0),
            anchor_in_child:  Vec3::new(0.0,  0.05, 0.0) },
        // Right arm
        JointDef { parent: SegmentId::Chest, child: SegmentId::UpperArmR,
            anchor_in_parent: Vec3::new(0.14,  0.10, 0.0),
            anchor_in_child:  Vec3::new( 0.0,  0.15, 0.0) },
        JointDef { parent: SegmentId::UpperArmR, child: SegmentId::LowerArmR,
            anchor_in_parent: Vec3::new(0.0, -0.15, 0.0),
            anchor_in_child:  Vec3::new(0.0,  0.13, 0.0) },
        JointDef { parent: SegmentId::LowerArmR, child: SegmentId::HandR,
            anchor_in_parent: Vec3::new(0.0, -0.13, 0.0),
            anchor_in_child:  Vec3::new(0.0,  0.05, 0.0) },
        // Left leg
        JointDef { parent: SegmentId::Pelvis, child: SegmentId::UpperLegL,
            anchor_in_parent: Vec3::new(-0.10, -0.10, 0.0),
            anchor_in_child:  Vec3::new(  0.0,  0.20, 0.0) },
        JointDef { parent: SegmentId::UpperLegL, child: SegmentId::LowerLegL,
            anchor_in_parent: Vec3::new(0.0, -0.20, 0.0),
            anchor_in_child:  Vec3::new(0.0,  0.19, 0.0) },
        JointDef { parent: SegmentId::LowerLegL, child: SegmentId::FootL,
            anchor_in_parent: Vec3::new(0.0, -0.19, 0.0),
            anchor_in_child:  Vec3::new(0.0,  0.04, 0.0) },
        // Right leg
        JointDef { parent: SegmentId::Pelvis, child: SegmentId::UpperLegR,
            anchor_in_parent: Vec3::new(0.10, -0.10, 0.0),
            anchor_in_child:  Vec3::new( 0.0,  0.20, 0.0) },
        JointDef { parent: SegmentId::UpperLegR, child: SegmentId::LowerLegR,
            anchor_in_parent: Vec3::new(0.0, -0.20, 0.0),
            anchor_in_child:  Vec3::new(0.0,  0.19, 0.0) },
        JointDef { parent: SegmentId::LowerLegR, child: SegmentId::FootR,
            anchor_in_parent: Vec3::new(0.0, -0.19, 0.0),
            anchor_in_child:  Vec3::new(0.0,  0.04, 0.0) },
    ]
}

// Y offsets derived from joint anchor chain so every joint starts with zero constraint error.
// Chain: foot=0.04 → lower_leg=0.27 (+0.23) → upper_leg=0.66 (+0.39) → pelvis=0.96 (+0.30)
//        pelvis → lumbar=1.15 (+0.19) → chest=1.38 (+0.23) → neck=1.58 (+0.20) → head=1.74 (+0.16)
//        chest shoulder → upper_arm=1.33 → lower_arm=1.05 → hand=0.87
fn segment_y_offset(seg: SegmentId) -> f64 {
    match seg {
        SegmentId::FootL | SegmentId::FootR                        => 0.04,
        SegmentId::LowerLegL | SegmentId::LowerLegR               => 0.27,
        SegmentId::UpperLegL | SegmentId::UpperLegR               => 0.66,
        SegmentId::Pelvis                                          => 0.96,
        SegmentId::Lumbar                                          => 1.15,
        SegmentId::Chest                                           => 1.38,
        SegmentId::Neck                                            => 1.58,
        SegmentId::Head                                            => 1.74,
        SegmentId::UpperArmL | SegmentId::UpperArmR               => 1.33,
        SegmentId::LowerArmL | SegmentId::LowerArmR               => 1.05,
        SegmentId::HandL | SegmentId::HandR                       => 0.87,
        SegmentId::_Count                                          => unreachable!(),
    }
}

// X offsets match joint anchor X values so constraints start satisfied.
// Chest shoulder anchor at x=±0.14, so arm segments share x=±0.14.
fn segment_x_offset(seg: SegmentId) -> f64 {
    match seg {
        SegmentId::UpperArmL | SegmentId::LowerArmL | SegmentId::HandL   => -0.14,
        SegmentId::UpperArmR | SegmentId::LowerArmR | SegmentId::HandR   =>  0.14,
        SegmentId::UpperLegL | SegmentId::LowerLegL | SegmentId::FootL   => -0.10,
        SegmentId::UpperLegR | SegmentId::LowerLegR | SegmentId::FootR   =>  0.10,
        _ => 0.0,
    }
}

fn ang(a: SegmentId, b: SegmentId, axis: Vec3, lo: f64, hi: f64) -> AngularLimitConstraint {
    AngularLimitConstraint::new(a.idx(), b.idx(), axis, lo, hi)
}

/// One constraint per axis per joint — three axes (Z sagittal, X lateral, Y twist).
/// Angles in radians; measured as relative rotation of child w.r.t. parent.
fn build_angular_constraints() -> Vec<AngularLimitConstraint> {
    use SegmentId::*;
    let x = Vec3::x();
    let y = Vec3::y();
    let z = Vec3::z();
    vec![
        // ── Spine chain ────────────────────────────────────────────────────
        // Pelvis → Lumbar
        ang(Pelvis, Lumbar, z, -0.35, 0.45),
        ang(Pelvis, Lumbar, x, -0.25, 0.25),
        ang(Pelvis, Lumbar, y, -0.30, 0.30),
        // Lumbar → Chest
        ang(Lumbar, Chest, z, -0.20, 0.30),
        ang(Lumbar, Chest, x, -0.20, 0.20),
        ang(Lumbar, Chest, y, -0.20, 0.20),
        // Chest → Neck
        ang(Chest, Neck, z, -0.40, 0.50),
        ang(Chest, Neck, x, -0.30, 0.30),
        ang(Chest, Neck, y, -0.50, 0.50),
        // Neck → Head
        ang(Neck, Head, z, -0.20, 0.35),
        ang(Neck, Head, x, -0.20, 0.20),
        ang(Neck, Head, y, -0.35, 0.35),

        // ── Arms ───────────────────────────────────────────────────────────
        // Chest → UpperArm (ball-and-socket)
        // Z forward-flexion limit is -2.97 rad (~170°). Stopping before -π = -3.14
        // is critical: at -π the quaternion error representation wraps sign and the
        // spring drives the arm in the wrong direction, causing divergent oscillation.
        ang(Chest, UpperArmL, z, -2.97, 1.57),
        ang(Chest, UpperArmL, x, -1.57, 1.57),
        ang(Chest, UpperArmL, y, -1.57, 1.57),
        ang(Chest, UpperArmR, z, -2.97, 1.57),
        ang(Chest, UpperArmR, x, -1.57, 1.57),
        ang(Chest, UpperArmR, y, -1.57, 1.57),
        // UpperArm → LowerArm (elbow — hinge)
        // Lower bound is -0.40 (not 0) so the lighter lower arm can lag the upper
        // arm slightly during large swings without triggering the constraint.
        ang(UpperArmL, LowerArmL, z, -0.40, 2.20),
        ang(UpperArmL, LowerArmL, x, -0.15, 0.15),
        ang(UpperArmL, LowerArmL, y, -0.30, 0.30),
        ang(UpperArmR, LowerArmR, z, -0.40, 2.20),
        ang(UpperArmR, LowerArmR, x, -0.15, 0.15),
        ang(UpperArmR, LowerArmR, y, -0.30, 0.30),
        // LowerArm → Hand (wrist)
        // Z limit is wide so the wrist never fires when the arm swings to large angles
        // while the hand spring holds near identity. A tight Z limit caused the lower
        // arm to oscillate, which propagated up and destabilized the upper arm.
        ang(LowerArmL, HandL, z, -3.14, 3.14),
        ang(LowerArmL, HandL, x, -0.40, 0.40),
        ang(LowerArmL, HandL, y, -0.30, 0.30),
        ang(LowerArmR, HandR, z, -3.14, 3.14),
        ang(LowerArmR, HandR, x, -0.40, 0.40),
        ang(LowerArmR, HandR, y, -0.30, 0.30),

        // ── Legs ───────────────────────────────────────────────────────────
        // Pelvis → UpperLeg (hip — ball-and-socket)
        ang(Pelvis, UpperLegL, z, -0.52, 0.70),
        ang(Pelvis, UpperLegL, x, -0.35, 0.35),
        ang(Pelvis, UpperLegL, y, -0.35, 0.35),
        ang(Pelvis, UpperLegR, z, -0.52, 0.70),
        ang(Pelvis, UpperLegR, x, -0.35, 0.35),
        ang(Pelvis, UpperLegR, y, -0.35, 0.35),
        // UpperLeg → LowerLeg (knee — hinge, bends in positive Z)
        ang(UpperLegL, LowerLegL, z,  0.00, 1.50),
        ang(UpperLegL, LowerLegL, x, -0.10, 0.10),
        ang(UpperLegL, LowerLegL, y, -0.10, 0.10),
        ang(UpperLegR, LowerLegR, z,  0.00, 1.50),
        ang(UpperLegR, LowerLegR, x, -0.10, 0.10),
        ang(UpperLegR, LowerLegR, y, -0.10, 0.10),
        // LowerLeg → Foot (ankle)
        ang(LowerLegL, FootL, z, -0.40, 0.50),
        ang(LowerLegL, FootL, x, -0.25, 0.25),
        ang(LowerLegL, FootL, y, -0.15, 0.15),
        ang(LowerLegR, FootR, z, -0.40, 0.50),
        ang(LowerLegR, FootR, x, -0.25, 0.25),
        ang(LowerLegR, FootR, y, -0.15, 0.15),
    ]
}

pub struct PlayerSkeleton {
    pub bodies: Vec<RigidBody>,
    pub constraints: Vec<PositionalConstraint>,
    pub angular_constraints: Vec<AngularLimitConstraint>,
    pub body_type: BodyType,
}

impl PlayerSkeleton {
    pub fn new(body_type: BodyType, root: Vec3) -> Self {
        let mut bodies = Vec::with_capacity(SEGMENT_COUNT);

        for seg in SegmentId::all() {
            let def = segment_def(body_type, seg);
            let mut body = RigidBody::new(def.mass, def.inertia_tensor());
            body.position = root + Vec3::new(
                segment_x_offset(seg),
                segment_y_offset(seg),
                0.0,
            );
            bodies.push(body);
        }

        let mut constraints = Vec::with_capacity(JOINT_COUNT);
        for jd in joint_defs() {
            let pa = jd.parent.idx();
            let ca = jd.child.idx();
            let mut c = PositionalConstraint::new(pa, ca, 0.0, 0.0);
            c.local_anchor_a = jd.anchor_in_parent;
            c.local_anchor_b = jd.anchor_in_child;
            constraints.push(c);
        }

        let angular_constraints = build_angular_constraints();

        Self { bodies, constraints, angular_constraints, body_type }
    }

    pub fn total_mass(&self) -> f64 {
        self.bodies.iter().map(|b| b.mass).sum()
    }

    pub fn centre_of_mass(&self) -> Vec3 {
        let total = self.total_mass();
        self.bodies.iter().map(|b| b.position * b.mass).sum::<Vec3>() / total
    }
}
