use physics::rigid_body::RigidBody;
use physics::constraints::PositionalConstraint;
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

pub struct PlayerSkeleton {
    pub bodies: Vec<RigidBody>,
    pub constraints: Vec<PositionalConstraint>,
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

        let mut constraints = Vec::with_capacity(JOINT_COUNT * 2);
        for jd in joint_defs() {
            let pa = jd.parent.idx();
            let ca = jd.child.idx();
            let mut c1 = PositionalConstraint::new(pa, ca, 0.0, 1e-6);
            c1.local_anchor_a = jd.anchor_in_parent;
            c1.local_anchor_b = jd.anchor_in_child;
            let mut c2 = PositionalConstraint::new(pa, ca, 0.0, 1e-6);
            c2.local_anchor_a = jd.anchor_in_parent;
            c2.local_anchor_b = jd.anchor_in_child;
            constraints.push(c1);
            constraints.push(c2);
        }

        Self { bodies, constraints, body_type }
    }

    pub fn total_mass(&self) -> f64 {
        self.bodies.iter().map(|b| b.mass).sum()
    }

    pub fn centre_of_mass(&self) -> Vec3 {
        let total = self.total_mass();
        self.bodies.iter().map(|b| b.position * b.mass).sum::<Vec3>() / total
    }
}
