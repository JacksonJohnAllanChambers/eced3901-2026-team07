# ECED3901 Team 07 — Navigation Overview

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Arena Layout and Geometry](#arena-layout-and-geometry)
3. [Navigation Strategies](#navigation-strategies)
4. [Localization (AMCL)](#localization-amcl)
5. [Odometry and Sensor Fusion (EKF)](#odometry-and-sensor-fusion-ekf)
6. [Costmap and Planner Tuning](#costmap-and-planner-tuning)
7. [Controller Tuning (DWB)](#controller-tuning-dwb)
8. [Simulation vs. Real Robot Differences](#simulation-vs-real-robot-differences)
9. [Launch File Architecture](#launch-file-architecture)
10. [Static Map Generation](#static-map-generation)
11. [Diagnostic Tooling](#diagnostic-tooling)
12. [Potential Issues and Risks](#potential-issues-and-risks)
13. [File Reference](#file-reference)

---

## System Architecture

### TF Frame Chain

```
map  →  odom  →  base_footprint  →  base_link  →  lidar_link
 │        │            │
 │        │            └── Published by Gazebo diff_drive plugin
 │        └── Published by AMCL (map→odom correction)
 └── Static frame (world origin)
```

- **AMCL** publishes the `map → odom` transform by matching lidar scans against the static map.
- **Gazebo diff_drive plugin** publishes `odom → base_footprint` using encoder-based odometry.
- **EKF** (`robot_localization`) runs but with `publish_tf: false` — it does NOT publish TF to avoid a dual-publisher conflict with the diff_drive plugin. Both were publishing `odom → base_footprint`, causing TF flickering, AMCL convergence failures, and robot teleporting in RViz.
- **URDF** defines `base_footprint → base_link → lidar_link` as static transforms.

### Nav2 Stack Components

| Component | Plugin | Purpose |
|-----------|--------|---------|
| Planner | `NavfnPlanner` (A*) | Global path planning on the costmap |
| Controller | `DWBLocalPlanner` | Local trajectory following with obstacle avoidance |
| Localization | AMCL | Particle-filter localization against a known map |
| Recovery | Spin, Backup, Wait | Behaviours triggered when the robot gets stuck |
| Waypoint Follower | `WaitAtWaypoint` | Sequential waypoint execution (pause = 0s) |
| Map Server | `map_server` | Serves the pre-built static arena map |

### Node Graph (Arena Challenge)

```
arena_gazebo.launch.py              arena_amcl.launch.py
├── gazebo_server                   ├── nav2_bringup (AMCL + map_server + planner + controller + BT)
├── gazebo_client                   └── rviz2 (optional)
├── robot_state_publisher
├── ekf_filter_node                 Navigation script (choose one):
└── set_gazebo_start_pose           ├── arena_nav_wp.py    (Nav2 waypoint follower)
                                    └── arena_direct_nav.py (direct cmd_vel control)
```

---

## Arena Layout and Geometry

### Dimensions

- **Arena size**: 14 ft × 14 ft (4.267 m × 4.267 m)
- **Wall thickness**: 0.15 m (all perimeter, divider, and zigzag walls)
- **Robot width**: ~0.22 m (`robot_radius: 0.11`)

### Lane Structure

```
  x=0ft         x=4ft        x=10ft       x=14ft
  (0.000m)      (1.219m)     (3.048m)     (4.267m)
    │              │            │              │
    │  Left        │   Open     │    Right     │
    │  Coastal     │   Waters   │    Coastal   │
    │  (zigzag)    │  (pirate   │    (zigzag)  │
    │              │   zone)    │              │
    │              │            │              │
```

### Zigzag Wall Layout (Left Coastal, x = 0–4 ft)

```
y=14ft ┌──────────────────────────┐  Target Ports (top)
       │  PORT                    │
y=12ft │                          │
       │                          │
y=10ft │  ████████░░░░░░░░░░░░░░ │  Wall 3 from LEFT  → gap on RIGHT
       │                          │
y=7ft  │  ░░░░░░░░░░░░████████░ │  Wall 2 from RIGHT → gap on LEFT
       │                          │
y=4ft  │  ████████░░░░░░░░░░░░░░ │  Wall 1 from LEFT  → gap on RIGHT
       │                          │
y=1ft  │  DEPLOY                  │  Deployment Zones (bottom)
y=0ft  └──────────────────────────┘
       x=0ft                  x=4ft

████ = wall (2 ft / 0.610 m long)
░░░░ = gap  (≈0.534 m usable width after wall thickness)
```

Right coastal is the mirror image (walls from RIGHT/LEFT alternate in reverse).

### Critical Gap Dimensions

- **Lane width** (inner face to inner face): ~1.07 m
- **Zigzag wall length**: 0.610 m (2 ft) + 0.15 m thick
- **Usable gap width**: ~0.534 m
- **Robot clearance per side**: ~0.157 m (robot 0.22 m through 0.534 m gap)
- This is extremely tight — only ~7 cm margin on each side after inflation.

### Start Positions

| Position | x (ft) | y (ft) | x (m) | y (m) | Heading (sim) | Heading (real) |
|----------|--------|--------|-------|-------|----------------|----------------|
| `left_coastal` | 3 | 1 | 0.914 | 0.305 | π/2 (north) | π (west) |
| `left_open` | 5 | 1 | 1.524 | 0.305 | π/2 (north) | π (west) |
| `right_open` | 9 | 1 | 2.743 | 0.305 | π/2 (north) | π (west) |
| `right_coastal` | 11 | 1 | 3.353 | 0.305 | π/2 (north) | π (west) |

---

## Navigation Strategies

Two independent navigation approaches exist for the arena challenge:

### Strategy 1: Nav2 Waypoint Follower (`arena_nav_wp.py`)

Uses the full Nav2 stack (planner + controller + costmaps) with `BasicNavigator.followWaypoints()`.

**How it works:**
1. Sets the initial AMCL pose via `setInitialPose()`
2. Waits for Nav2 lifecycle activation (`waitUntilNav2Active()`)
3. Sends a list of `PoseStamped` waypoints to the waypoint follower
4. Nav2 plans a global path and the DWB controller follows it
5. Forward route → pause at port → return route

**Coastal route (per side):**
- 10 waypoints forward + port, 10 waypoints return + home
- Zigzag pattern: align to gap → pass through wall → cross to opposite gap
- Waypoints placed before, at, and after each wall

**Open water route:**
- Straight north/south through open waters (6 waypoints each direction)
- No pirate-zone avoidance logic

**Gap centre coordinates used (waypoint navigator):**

| Gap | Left Coastal | Right Coastal |
|-----|-------------|---------------|
| Near divider | 0.99 m | 3.28 m |
| Near outer wall | 0.23 m | 4.04 m |
| Lane centre | 0.61 m | 3.66 m |

### Strategy 2: Direct cmd_vel Control (`arena_direct_nav.py`)

Bypasses the Nav2 planner and controller entirely. Uses simple rotate-then-drive primitives with odom-based position tracking calibrated against the AMCL TF at startup.

**How it works:**
1. Publishes initial pose to AMCL, waits for convergence (up to 60 attempts, re-publishes pose every 15 attempts)
2. Calibrates odom-to-map transform: `map_yaw = odom_yaw + yaw_offset`, `map_xy = R(yaw_offset) @ odom_xy + (x_offset, y_offset)`
3. Position feedback from raw `/odom` topic (Gazebo diff_drive), NOT from TF — immune to AMCL particle filter jitter
4. Heading from odom yaw + calibrated offset (smooth, continuous)
5. Gate waypoints use y-crossing detection (passthrough — robot never stops at gaps)
6. Destination waypoints use distance tolerance with overshoot detection

**Gate passthrough system:**
- Gate waypoints are `(x, y, 'north')` or `(x, y, 'south')` tuples
- As the robot approaches a wall gap, it checks if its y coordinate has crossed the wall y-level
- Once `y > wall_y` (northbound) or `y < wall_y` (southbound), the gate is considered crossed
- The robot never orbits or stops at a gap — it drives straight through
- If the robot is already past a gate when starting (e.g. from odometry drift), it skips without stopping

**Odom-based position tracking:**
- Raw `/odom` from Gazebo diff_drive provides smooth, continuous position at ~30 Hz
- One-time calibration at startup computes rotation matrix + translation offset to convert odom→map coordinates
- Recalibration only at destination arrivals (not gates), with a 45° yaw change sanity check to reject AMCL jitter
- In testing, odom drift over a full 2-minute round trip is ~1–6° yaw, well within tolerance

**Key parameters:**
- Linear speed: 0.1 m/s
- Rotation speed: 0.2 rad/s (max)
- Yaw tolerance: 0.05 rad (~2.9°)
- Distance tolerance: 0.10 m
- Steering: proportional gain = 0.6, max = 0.3 rad/s, deadband = 0.03 rad
- Stuck detection: 6s without 2cm odom movement → skip waypoint
- Per-waypoint timeout: 35s

**Gap centre coordinates used (direct navigator):**

| Gap | Left Coastal | Right Coastal |
|-----|-------------|---------------|
| Near divider (walls 1 & 3) | 0.877 m | 3.390 m |
| Near outer wall (wall 2) | 0.342 m | 3.925 m |

These are the true geometric centres of the 0.534 m wide gaps, computed from the world file wall positions.

### Strategy Comparison

| Aspect | Nav2 Waypoints | Direct cmd_vel |
|--------|---------------|----------------|
| Obstacle avoidance | Real-time via costmap | None (odom-based dead reckoning) |
| Path smoothness | DWB-optimized curves | Rotate-drive segments |
| Recovery behaviour | Spin/backup/wait | Stuck + overshoot detection |
| Dependency | Full Nav2 stack | Only AMCL (for initial calibration) + raw `/odom` |
| Gap navigation | Costmap inflation can block narrow gaps | Gate passthrough — crosses gap centres within 1–3 mm |
| Heading source | Nav2 controller | Odom yaw + calibrated offset |
| Position source | AMCL (can jitter significantly) | Raw odom + calibrated rotation matrix (smooth, continuous) |
| Complexity | Higher (more failure modes) | Lower (predictable) |
| Typical round-trip time | Unknown (untested on this arena) | ~120 s for full 8-waypoint round trip |

> **Recommended for arena challenge:** Direct cmd_vel. In simulation testing, it consistently navigates all 6 gaps (3 forward + 3 return) with 1–3 mm accuracy at gap crossings. The Nav2 approach risks being blocked by costmap inflation in the 53 cm wide gaps (with only 2 cm inflation radius, the planner works but the DWB controller with 0.02 BaseObstacle weight provides almost no reactive obstacle avoidance).

---

## Localization (AMCL)

AMCL is the sole localization method for the arena challenge (no SLAM).

### Key Parameters

| Parameter | Simulation | Real Robot | Rationale |
|-----------|-----------|------------|-----------|
| `alpha1–5` (motion noise) | 0.01 | 0.2 | Real robot has much higher odometry uncertainty |
| `max_particles` | 2000 | 5000 | More particles compensate for real-world noise |
| `min_particles` | 500 | 1000 | Higher floor for robustness |
| `recovery_alpha_fast` | 0.0 | 0.1 | Sim: disabled. Real: enables fast recovery |
| `recovery_alpha_slow` | 0.0 | 0.05 | Sim: disabled. Real: enables slow recovery |
| `update_min_a` | 0.4 rad | 0.05 rad | Real: update more frequently with less rotation |
| `update_min_d` | 0.15 m | 0.05 m | Real: update more frequently with less translation |
| `initial_pose yaw` | 1.5708 (north) | 3.1416 (west) | Different physical starting orientation |
| `laser_model_type` | likelihood_field | likelihood_field | Standard model for structured environments |
| `laser_max_range` | 5.0 m | 5.0 m | Full arena diagonal ≈ 6.0 m |
| `set_initial_pose` | True | True | Auto-set pose at startup |

### Initial Pose Handling

Both navigation scripts also publish the initial pose independently:
- `arena_nav_wp.py`: via `navigator.setInitialPose()`
- `arena_direct_nav.py`: via `/initialpose` topic (published 5 times with 100ms spacing)

The direct navigator additionally waits up to 30s for AMCL convergence by comparing the TF pose against the expected start position (tolerance: 0.15 m XY, 0.3 rad yaw). It re-publishes the initial pose every 15 attempts to recover from AMCL missing the first publication. If AMCL does not converge, the navigator **aborts** rather than proceeding with a bad calibration.

---

## Odometry and Sensor Fusion (EKF)

### Arena Configuration (`config/ekf_arena.yaml`)

The arena EKF fuses **wheel odometry + IMU yaw angular velocity**.

```
Input 1: wheel/odometry
  Fused: x position, y position, yaw position, x velocity, yaw velocity

Input 2: imu/data
  Fused: yaw angular velocity (gyro z-axis) only
  NOT fused: orientation (Gazebo reports absolute world yaw, conflicts with odom frame)
```

**Critical: `publish_tf: false`**

The EKF does NOT publish the `odom → base_footprint` transform. This is intentional — the Gazebo diff_drive plugin already publishes this transform via `<publish_odom_tf>true</publish_odom_tf>`. If both publish the same TF, the transform flickering causes:
- AMCL convergence failures (particle filter gets unstable input)
- Robot teleporting in RViz (TF alternates between two different values)
- Navigator receiving garbage position data

**Topic mismatch note:** The EKF subscribes to `wheel/odometry`, but the diff_drive plugin publishes to `/odom`. These are different topics. Unless something remaps `wheel/odometry` → `/odom`, the EKF receives no wheel odom data. The direct navigator uses raw `/odom` directly, bypassing the EKF entirely.

### Generic Configuration (`config/ekf.yaml`)

Used for non-arena labs. Fuses both wheel odometry AND IMU:
- Odom: x, y position
- IMU: yaw angular velocity only
- Base frame: `base_link`

---

## Costmap and Planner Tuning

### Inflation Strategy

The costmaps use an extremely aggressive configuration to allow passage through narrow gaps:

| Parameter | Arena Value | Default/Lab Value | Impact |
|-----------|------------|-------------------|--------|
| `inflation_radius` | **0.02 m** | 0.40–0.95 m | Only 2 cm beyond lethal obstacles |
| `cost_scaling_factor` | **15.0** | 3.0 | Very steep cost falloff — free space starts almost immediately |
| `robot_radius` | 0.11 m | 0.11 m | Half-width of the TurtleBot3 Burger |
| `resolution` | 0.025 m | 0.05 m | Finer grid for narrow gap accuracy |

**Why this is necessary:** The gaps are ~0.534 m wide and the robot is 0.22 m wide, leaving only ~0.157 m per side. With a standard inflation radius (0.4 m), NavFn would see the gaps as completely blocked and refuse to plan through them.

**Trade-off:** With nearly zero inflation, the planner will happily plan paths that graze walls. Any localization error > ~5 cm could cause a wall collision.

### Global Planner (NavFn)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `use_astar` | true | A* is more efficient in maze-like environments than Dijkstra |
| `tolerance` | 0.1 m | Tight tolerance (was 0.5) — ensures waypoints near walls are reachable |
| `allow_unknown` | false | Full map is available, no need to plan into unknown space |
| `expected_planner_frequency` | 5 Hz | Faster replanning for dynamic adjustments |

### Costmap Layers

Both local and global costmaps use:
1. **Static layer** (global only) — pre-built arena map
2. **Obstacle layer** — real-time lidar obstacle detection
3. **Inflation layer** — cost inflation around obstacles

The local costmap uses a **3 m × 3 m rolling window** (arena is only 4.267 m wide, so this covers most of the robot's surroundings).

---

## Controller Tuning (DWB)

### Velocity Limits

| Parameter | Value | Note |
|-----------|-------|------|
| `max_vel_x` | 0.18 m/s | TurtleBot3 Burger max is 0.22; conservative limit |
| `max_vel_theta` | 0.8 rad/s | Agile turning for tight corridor navigation |
| `controller_frequency` | 10 Hz | 5× faster than default (2 Hz) for responsiveness |

### Trajectory Sampling

| Parameter | Value |
|-----------|-------|
| `vx_samples` | 20 |
| `vtheta_samples` | 30 (increased for tight gaps) |
| `sim_time` | 1.5 s |

### DWB Critics and Weights

| Critic | Scale | Purpose |
|--------|-------|---------|
| `RotateToGoal` | 32.0 | Strong preference to face the goal |
| `PathAlign` | 32.0 | Stay aligned with the planned path |
| `PathDist` | 32.0 | Stay close to the planned path |
| `GoalAlign` | 24.0 | Align heading toward goal |
| `GoalDist` | 24.0 | Minimize distance to goal |
| `BaseObstacle` | **0.02** | Very weak obstacle avoidance bias |
| `Oscillation` | (default) | Dampen oscillatory behaviour |

### Goal Tolerances

| Parameter | Value |
|-----------|-------|
| `xy_goal_tolerance` | 0.15 m |
| `yaw_goal_tolerance` | 0.35 rad (~20°) |
| `required_movement_radius` | 0.15 m (progress checker) |
| `movement_time_allowance` | 15.0 s |

---

## Simulation vs. Real Robot Differences

| Aspect | Simulation | Real Robot |
|--------|-----------|------------|
| `use_sim_time` | True | False |
| AMCL `alpha1–5` | 0.01 | 0.2 |
| AMCL `max_particles` | 2000 | 5000 |
| AMCL recovery | Disabled (0.0) | Enabled (fast=0.1, slow=0.05) |
| AMCL update thresholds | 0.4 rad / 0.15 m | 0.05 rad / 0.05 m |
| Initial yaw | 1.5708 (north/+Y) | 3.1416 (west/−X) |
| EKF `publish_tf` | **false** (diff_drive publishes TF) | **true** (no diff_drive plugin) |
| EKF IMU fusion | Gyro z-axis angular velocity | Gyro z-axis angular velocity |
| Params file | `arena_nav2_params.yaml` | `arena_real_nav2_params.yaml` |
| Launch file | `arena_amcl.launch.py` | `arena_real_amcl.launch.py` |

The real robot parameters are more conservative: higher particle counts, faster update rates, and enabled particle recovery to handle real-world sensor noise and mechanical imprecision.

---

## Launch File Architecture

### Arena Simulation Workflow

```bash
# Terminal 1: Start Gazebo + EKF + robot model
ros2 launch eced3901 arena_gazebo.launch.py start:=left_coastal

# Terminal 2: Start Nav2 + AMCL + map server
ros2 launch eced3901 arena_amcl.launch.py

# Terminal 3: Run navigation (choose one)
ros2 run eced3901 arena_nav_wp.py --ros-args -p start:=left_coastal
# or
ros2 run eced3901 arena_direct_nav.py --ros-args -p start:=left_coastal
```

### Real Robot Workflow

```bash
# Terminal 1: Robot bringup (TurtleBot3 specific, not in this package)
# ... start robot, lidar, etc.

# Terminal 2: Start Nav2 with real params
ros2 launch eced3901 arena_real_amcl.launch.py

# Terminal 3: Run navigation
ros2 run eced3901 arena_nav_wp.py --ros-args -p start:=left_coastal
```

### Available Start Positions

The `start` argument accepts: `left_coastal`, `left_open`, `right_open`, `right_coastal`

Each sets:
- Gazebo spawn position (`set_gazebo_start_pose.py`)
- AMCL initial pose (in params YAML)
- Navigation waypoint selection (in nav scripts)

---

## Static Map Generation

The arena map is generated programmatically (`scripts/generate_arena_map.py`) rather than via SLAM, because the arena dimensions are known precisely.

### Map Parameters

| Parameter | Value |
|-----------|-------|
| Resolution | 0.02 m/pixel |
| Size | 239 × 239 pixels |
| Coverage | 4.767 m × 4.767 m (arena + 0.25 m margin) |
| Origin | (−0.25, −0.25, 0.0) |
| Format | PGM (P5 binary) + YAML metadata |

### Walls Encoded in Map

1. **4 perimeter walls** (bottom, top, left, right)
2. **2 lane dividers** (x = 4 ft, x = 10 ft — full height)
3. **3 left coastal zigzag walls** (at y = 4 ft, 7 ft, 10 ft)
4. **3 right coastal zigzag walls** (at y = 4 ft, 7 ft, 10 ft)

Areas outside the arena perimeter are marked as unknown (grey = 205).

---

## Diagnostic Tooling

### `nav_diag.py` — Navigation Stack Diagnostic Logger

A real-time diagnostic tool that snapshots the entire navigation state every second:

- **All TF transforms**: map→odom, odom→base_footprint, map→base_footprint, base_footprint→base_link, base_link→lidar_link
- **Odometry**: `/odom` and `/wheel/odometry` (position, velocity, frame IDs, message counts)
- **IMU**: `/imu/data` (yaw, gyro_z, frame ID)
- **Lidar**: `/scan` (ray count, valid rays, minimum range)
- **Optional drive test**: drives forward 0.15 m/s for 1s then rotates 0.5 rad/s for 1s, printing before/after snapshots

```bash
# Passive monitoring
ros2 run eced3901 nav_diag.py

# With drive test
ros2 run eced3901 nav_diag.py --ros-args -p drive_test:=true
```

---

## Potential Issues and Risks

### 1. Gap Coordinate Disagreement Between Navigation Scripts

**Severity: Low** (was Medium — direct navigator now uses true geometric centres)

The two navigation scripts use different gap centre coordinates:

| Gap | `arena_direct_nav.py` | `arena_nav_wp.py` | Note |
|-----|----------------------|-------------------|------|
| Left coastal, near divider | 0.877 m | 0.99 m | Direct nav uses true centre |
| Left coastal, near outer | 0.342 m | 0.23 m | Direct nav uses true centre |
| Right coastal, near divider | 3.390 m | 3.28 m | Direct nav uses true centre |
| Right coastal, near outer | 3.925 m | 4.04 m | Direct nav uses true centre |

The direct navigator's coordinates are the geometric centres of each 0.534 m gap, computed from the world file. The waypoint navigator's coordinates are approximations. In simulation testing, the direct navigator achieves 1–3 mm accuracy at each gap crossing.

### 2. Extremely Aggressive Inflation Radius

**Severity: High** (Nav2 only — direct navigator bypasses costmaps)

The `inflation_radius` of 0.02 m means the costmap adds virtually no safety buffer around obstacles. The planner treats space just 2 cm from a wall as completely free.

- With a `robot_radius` of 0.11 m, the effective clearance check is only 0.13 m from wall centres
- Real-world AMCL localization drifts of 3–5 cm could cause the robot to contact walls
- The `cost_scaling_factor` of 15.0 makes the cost drop to nearly zero within millimetres of the lethal zone
- **Risk**: Wall collisions in tight gaps, especially on the real robot

### 3. BaseObstacle Critic Scale Too Low

**Severity: Medium** (Nav2 only)

The DWB `BaseObstacle.scale` is set to 0.02, while `PathDist` and `PathAlign` are at 32.0. This means the controller weights path-following **1600× more** than obstacle avoidance.

### 4. AMCL Particle Recovery Disabled in Simulation

**Severity: Low–Medium**

In `arena_nav2_params.yaml`, `recovery_alpha_fast` and `recovery_alpha_slow` are both 0.0.

- If AMCL's particle cloud converges to the wrong location, it cannot recover
- The arena has high symmetry (left/right coastal paths are mirrors)
- The real robot params correctly enable recovery (0.1 / 0.05)

### 5. No Pirate Zone Avoidance

**Severity: Low**

Open water routes navigate straight through the pirate zone. Neither script avoids obstacles in this zone.

### 6. EKF Topic Mismatch

**Severity: Low** (EKF is not critical — direct navigator uses raw `/odom`)

The EKF subscribes to `wheel/odometry` but the Gazebo diff_drive plugin publishes to `/odom`. Without a topic remap, the EKF receives no wheel odom data and `/odometry/filtered` is unreliable.

- The direct navigator bypasses this by subscribing directly to `/odom`
- For the real robot, the EKF topic should be configured to match the actual odom source
- A separate `ekf_arena_real.yaml` with `publish_tf: true` (since there's no diff_drive plugin on real hardware) and IMU gyro fusion would be advisable

### 7. Odom Drift on Real Robot

**Severity: Medium–High** (direct navigator specific)

The direct navigator relies on encoder-based odometry for position tracking. On a real robot:
- Wheel slip (carpet, uneven floors) degrades position accuracy
- Yaw drift is typically 1–3°/m for differential drive
- Over an 8 m round trip, the robot could drift 5–15 cm laterally
- The available clearance per side is only 15.7 cm

Mitigations:
- The initial calibration anchors the odom frame to the correct map position
- Recalibration at destination arrivals (with 45° sanity check) corrects for moderate drift
- For the real robot, enabling IMU gyro fusion in the EKF (with `publish_tf: true` and diff_drive TF disabled) would improve heading estimates

### 8. Hardcoded Waypoints Assume Perfect Arena Construction

**Severity: Medium**

All waypoint coordinates are computed from the Gazebo world file, assuming exact dimensions.

- Real arena walls may have 1–3 cm placement errors
- Gap widths could vary from the nominal 0.534 m
- For the real robot, manual measurement-based adjustments may be needed

### 9. Documentation Mismatch in Real Robot Params

**Severity: Low**

The comment in `arena_real_nav2_params.yaml` states `initial_pose yaw: 0.0` but the actual value is `yaw: 3.1416` (π, west).

---

## File Reference

### Parameter Files

| File | Purpose |
|------|---------|
| `params/arena_nav2_params.yaml` | Nav2 parameters for arena simulation (use_sim_time: True) |
| `params/arena_real_nav2_params.yaml` | Nav2 parameters for arena real robot (use_sim_time: False) |
| `params/nav2_params.yaml` | Generic Nav2 parameters for lab exercises |

### EKF Configuration

| File | Purpose |
|------|---------|
| `config/ekf_arena.yaml` | EKF for arena (wheel odom only, no IMU) |
| `config/ekf.yaml` | EKF for labs (wheel odom + IMU) |

### Navigation Scripts

| File | Purpose |
|------|---------|
| `scripts/arena_nav_wp.py` | Nav2 waypoint follower for arena challenge |
| `scripts/arena_direct_nav.py` | Direct cmd_vel navigator for arena challenge |
| `scripts/coastal_path_nav.py` | Nav2 waypoint follower for coastal corridor |
| `scripts/demo_inspection.py` | Simple 4-point square demo route |

### Utility Scripts

| File | Purpose |
|------|---------|
| `scripts/generate_arena_map.py` | Generates static arena map (PGM + YAML) |
| `scripts/nav_diag.py` | Real-time navigation diagnostic logger |
| `scripts/map_saver_node.py` | Saves SLAM-generated maps after delay |
| `scripts/set_gazebo_start_pose.py` | Sets robot spawn position in Gazebo |

### Launch Files (Arena)

| File | Purpose |
|------|---------|
| `launch/arena_gazebo.launch.py` | Gazebo + EKF + robot model for arena |
| `launch/arena_amcl.launch.py` | Nav2 + AMCL for arena simulation |
| `launch/arena_real_amcl.launch.py` | Nav2 + AMCL for arena real robot |
| `launch/arena_nav.launch.py` | Combined Gazebo + Nav2 (SLAM or AMCL) |

### Maps

| File | Purpose |
|------|---------|
| `maps/arena_map.pgm` + `arena_map.yaml` | Pre-built arena occupancy grid (0.02 m/px) |
| `maps/lab4_map.pgm` + `lab4_map.yaml` | Lab 4 map (0.05 m/px) |
| `maps/lab5_dt2_map.pgm` + `lab5_dt2_map.yaml` | Lab 5 DT2 map (0.05 m/px) |

### World Files

| File | Purpose |
|------|---------|
| `worlds/challenge_arena.world` | 14×14 ft arena with zigzag walls, ports, deploy zones |
| `worlds/coastal_path.world` | Coastal corridor test environment |
| `worlds/eced3901_lab4.world` | Lab 4 environment |
| `worlds/eced3901_lab6.world` | Lab 6 environment |
