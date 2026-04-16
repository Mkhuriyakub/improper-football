// Phase 2: articulated rigid body player with physics-based locomotion.
// Stub for now — implemented after ball physics are validated.

pub struct Player {
    pub mass: f64,
    pub position: [f64; 3],
}

impl Player {
    pub fn new(mass: f64) -> Self {
        Self { mass, position: [0.0; 3] }
    }
}
