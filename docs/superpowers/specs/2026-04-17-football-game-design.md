# Football Game Design Spec
**Date:** 2026-04-17
**Project:** improper-football
**Status:** HISTORICAL — superseded (2026-04-26) by the engine-first direction

> This is the original game-first plan. The project pivoted to building a
> biomechanically-faithful human-motion physics engine first (see README);
> the game-layer crates it describes were abandoned, and the tackle
> milestone was explicitly dropped as premature. Kept as a record of the
> starting point. Crate statuses and milestones below do NOT describe the
> current repo.

---

## 1. Vision

A fully physics-driven American football simulation. Every player movement, tackle, stumble, and catch emerges from a rigid body physics engine — no pre-authored animation clips. Physics leads animation at all times. The goal is realistic gameplay, motion, and appearance that a pure animation-state-machine approach (like Madden) cannot produce.

---

## 2. Game Modes

| Mode | Scope |
|------|-------|
| **Quick Play** | Ship first. Pick teams, play a full game. No persistence. |
| **Franchise** | Season, standings, playoffs, trades, draft. Add later. |
| **Career** | Control a single player from draft to retirement. Add later. |

All game modes share the same physics simulation and rules engine. Franchise and Career are data/progression layers built on top of Quick Play.

---

## 3. Core Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Teams | 11v11 | Full NFL simulation |
| Player animation | Full procedural physics (no clips) | Physics leads animation; emergent realism |
| Player control | Hybrid (human controls one player; AI assists rest) | Ratings and play-calling matter, not just stick skill |
| Camera | Behind-QB primary; other angles later | Traditional, immersive |
| Art style | Photorealistic 3D | glTF meshes, PBR materials, Bevy renderer |
| Platform | PC (Desktop) | Single target for v1 |
| Multiplayer | Single-player v1; multiplayer-ready architecture | Design interfaces to support netcode later |

---

## 4. Architecture

Three major layers, each with a single source of truth:

```
INPUT / AI DIRECTOR / GAME RULES
          ↓
    PLAYER CONTROLLER
    (3-layer: Play → Locomotion → PD Joint)
          ↓
    XPBD PHYSICS ENGINE
    (374 rigid bodies: 22 players × 17 segments + ball)
          ↓
    BEVY RENDERER
    (skeletal mesh skinning driven by physics transforms)
```

**Physics is the sole source of truth.** The renderer reads physics transforms. The rules engine observes physics events. Nothing overrides the physics engine.

### Crate Structure

| Crate | Status | Responsibility |
|-------|--------|---------------|
| `physics` | ✅ Complete | XPBD solver, rigid bodies, constraints, math |
| `ball` | ✅ Complete | Football aerodynamics (drag, Magnus, spiral) |
| `player` | 🟡 Stub | Expand to full articulated skeleton |
| `game_rules` | 🆕 New | Game state machine, scoring, clock, penalties |
| `ai` | 🆕 New | Play director, locomotion AI, coverage AI |
| `renderer` | 🆕 New | Bevy integration, mesh skinning, camera, HUD |
| `audio` | 🆕 New | Crowd, impact sounds, commentary |
| `ui` | 🆕 New | Play-call menu, pause, main menu |

---

## 5. Player Skeleton & Physics

Each of the 22 players on the field is a **17-segment articulated rigid body**:

**Segments:** Head · Chest · Pelvis · Upper arm ×2 · Lower arm ×2 · Hand ×2 · Upper leg ×2 · Lower leg ×2 · Foot ×2

**Joint types:**
- Ball-and-socket: Shoulders, hips (3 DOF)
- Hinge: Elbows, knees (1 DOF, anatomical limits)
- Universal: Ankles, wrists (2 DOF)
- Twist+Swing: Neck, spine (limited range)

**Mass distribution:** Realistic per body type. NFL lineman ~140kg, WR ~90kg. Heavier torso, lighter extremities. Inertia tensors match segment geometry.

**Collision shapes:** Capsule per segment (fast broad-phase, smooth contacts). Player–player contact transfers momentum through the constraint solver — tackles are physically emergent.

**PD Controller (per joint):**
```
τ = Kp × (θ_target − θ) − Kd × ω
```
Each joint has a target angle set by the locomotion controller. The PD controller computes torque to reach that angle without overshoot. Kp/Kd tuned per joint type and player role.

**Balance system:** Centre of mass computed every frame. Ankle and hip joints receive corrective torques to keep CoM over the support polygon. Disruption from contact causes physically realistic stumbling and falling.

---

## 6. Player AI & Control System

Three layers translate play-calling and input into joint torques each frame:

### Layer 1 — Play Director (Strategic)
Runs once per snap. Reads the called play and assigns every player a role and assignment:
- Receivers: route waypoints and timing
- Linemen: blocking assignments (target by position)
- Defenders: zone/man coverage assignments, blitz gaps
- Kickers/special teams: coverage lanes

### Layer 2 — Locomotion Controller (Tactical)
Runs every frame per player. Takes destination + intent, outputs desired velocity and body orientation.
- **Human-controlled player:** Left stick → desired velocity vector
- **AI players:** Pathfind to waypoints, react to ball position, obstacle avoidance
- **Hybrid assist:** Off-screen AI defenders rubber-band toward ball; catches and blocks resolved by player ratings when human isn't directly controlling

### Layer 3 — PD Joint Controller (Physical)
Runs every physics substep. Maps desired velocity → target joint angles for all 17 joints. Sub-components:
- **Stride generator:** Speed → stride length and frequency targets for leg joints
- **Upper body pose:** Role-specific postures (QB throwing stance, blocker lean, receiver hand position)
- **Contact response:** Incoming collision impulse disrupts balance targets → emergent stumble/fall

### QB Throw (Special Case)
Human input: right stick (aim direction) + hold/release button (throw power). Maps directly to the ball's initial velocity vector and spin parameters in the existing aerodynamics engine (`ball` crate). QB throwing arm joint angles are driven by release parameters — arm follows through physically. Pump fakes and scramble-throws work naturally.

---

## 7. Game Rules Engine

A state machine that governs every phase of the game. **Observes** the physics world — never overrides it.

### States
`PREGAME/KICKOFF → PLAY_CALL → PRE_SNAP → LIVE_PLAY → PLAY_RESULT → UPDATE_STATE → (loop)`

### Tracked State
- Quarter (1–4), game clock, play clock (40s)
- Down (1st–4th), yards to first down
- Ball position (yard line, hash mark)
- Score (home/away), possession
- Timeouts remaining (3 per half per team)

### Play Termination Conditions
Physics engine reports per-frame contact events. The rules engine checks:

| Condition | Physics Check |
|-----------|---------------|
| **Down by contact** | Any segment other than hands/feet contacts ground while a defender is in contact with ball carrier. Qualifying segments: knee, shin, forearm, elbow, hip, back, shoulder, head. |
| **Out of bounds** | Ball carrier's body exits field boundary |
| **Incomplete pass** | Ball contacts ground without catch |
| **Touchdown** | Ball crosses goal line in possession |
| **Interception** | Defender catches forward pass |
| **Fumble** | Ball separates from carrier and is recovered |
| **Safety** | Ball carrier downed in own end zone |

### Scoring
- Touchdown: 6 pts
- Extra point (kick): 1 pt
- Two-point conversion: 2 pts
- Field goal: 3 pts
- Safety: 2 pts (awarded to defense)

### Overtime Rules (2022+ NFL)
- Both teams guaranteed at least one possession
- Team A scores TD → Team B gets possession. TD: sudden death continues. FG or stop: Team A wins.
- Team A scores FG → Team B gets possession. TD: Team B wins. FG: sudden death. Stop: Team A wins.
- Team A fails → Team B gets possession. Any score wins. Stop: sudden death.
- Regular season: 10-minute period, tie possible
- Playoffs: 15-minute periods, repeat until winner

### Quick Play v1 Scope
Full 4-quarter game · All scoring types · Kickoffs · Punts · Basic penalty flags (false start, holding, pass interference) · Coin toss · NFL overtime rules

---

## 8. Rendering Pipeline

### Physics → Mesh Skinning
Each of the 17 rigid body segments per player produces a world-space transform every frame. These feed directly into Bevy's skeletal mesh skinning as bone transforms. No keyframes — physics is the animation.

### Bevy Render Stack
- **PBR materials:** Physically-based rendering for uniforms, helmets, skin, turf. Jersey numbers and team colors as material parameters.
- **Lighting:** Stadium floodlights (baked) + dynamic directional light. Day/night modes. Cascaded shadow maps for field coverage.
- **Post-processing:** HDR pipeline, bloom, tone mapping (Bevy built-in).
- **Crowd:** Instanced billboard sprites at distance. Performant, adds atmosphere.
- **Scoreboard:** Static mesh updated by rules engine each play.

### Ball Rendering
Football mesh is a child entity of the ball's rigid body transform. Quaternion orientation from the aerodynamics engine drives rotation directly — spiral spin is visually accurate automatically.

### Asset Pipeline
- Player meshes: rigged humanoid glTF models. Skeleton hierarchy must match 17-segment physics skeleton. Mixamo base meshes as starting point (free, rigged, retargetable).
- Field/stadium: glTF static meshes
- Textures: KTX2/Basis Universal compression

### Performance Targets
| Metric | Target |
|--------|--------|
| Frame rate | 60 fps |
| Physics bodies | 374 (22 × 17 + ball) |
| Physics substeps/frame | 4–8 |
| Draw calls | < 500 |

### Camera System
- **Behind-QB (primary):** Spring-arm follow. Zooms out on deep throws. Rotates with QB facing.
- **Ball-follow (in flight):** Tracks ball arc on deep throws, pulls back to show receiver and coverage.
- **Instant replay:** Free cinematic camera, 0.1× slow motion, 10-second rewind buffer.
- Additional camera angles designed in later iterations.

---

## 9. Input

### Gamepad (Primary)
| Input | Action |
|-------|--------|
| L-Stick | Move player / QB scramble |
| R-Stick | Aim throw direction |
| Hold A/X | Charge throw power |
| Release A/X | Throw |
| B/○ | Dive / slide |
| Y/△ | Spike / call timeout |
| LT/L2 | Sprint |
| RT/R2 | Juke / spin (ball carrier) |
| D-Pad | Switch controlled player |
| Start | Pause / play call menu |

### KB+M Fallback
| Input | Action |
|-------|--------|
| WASD | Move |
| Mouse | Aim throw |
| Hold Space | Charge throw power |
| 1–5 | Target receiver |
| Shift | Sprint |
| Tab | Switch player |

### Play-Call Menu
Radial menu on gamepad, grid layout on KB+M. Organized: Formation → Play type → Specific play. Route diagram shown on hover. Pre-snap window before play clock starts.

---

## 10. HUD

TV broadcast-style. All elements non-intrusive.

### Score Bug (top center)
A single centered unit:
```
[ NYG logo | 14  ]  [ 12:47        ]  [  21 | SF logo ]
[ • • ○   TO    ]  [ 2nd Quarter  ]  [  • • •   TO   ]
                    [ 2nd & 7 · SF32]
```
Team logo + score + timeout dots on each side. Clock + quarter + down/distance centered between them.

### Play Clock
Floats top-right, below score bug.

### In-Play Elements
- Yellow first-down line on field (rendered in 3D space)
- Blue line of scrimmage
- Yellow indicator above controlled player
- Receiver button prompts (float over receivers when QB is aiming)
- Throw power meter (right edge, appears when aiming)
- Field position strip (bottom center, mini-map of yard line)

---

## 11. Audio

### Crowd & Atmosphere
Dynamic crowd noise responding to game state. Volume/intensity rises on big plays, quiet on incompletions, roars on touchdowns. PA announcer and band stings driven by rules engine events.

### Impact & Physics Sounds
Collision impulse magnitude from physics engine drives hit intensity. Distinct sounds for: tackle, pad crack, helmet contact, footsteps, ball catch, ball hitting turf. Spatial audio — sounds positioned relative to camera.

### Commentary
Pre-recorded lines triggered by game state (down, score situation, play outcome). Canned but contextual for v1.

### Tech
Bevy built-in audio + `kira` crate for dynamic mixing. Spatial audio for positional sounds.

---

## 12. Tackle Simulation Milestone

Before integrating full player AI, build a sandbox scene with two physics bodies (one ball carrier, one defender) to validate tackle physics. This is the first observable proof-of-concept for the physics-driven player system. Target: two rigid body skeletons collide, balance is disrupted, emergent fall occurs — no scripting.

---

## 13. Out of Scope (v1)

- Online multiplayer (architecture must support it; netcode not implemented)
- Franchise mode persistence
- Career mode
- Custom playbooks beyond a base set
- Player customization / appearance editor
- Multiple camera angle selection in-game
- Full penalty system (basic flags only)
- Weather effects
