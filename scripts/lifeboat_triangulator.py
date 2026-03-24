#!/usr/bin/env python3
"""
===================================================================
Lifeboat Triangulator — 2D heatmap-based lifeboat position estimator
===================================================================
Accumulates bearing observations (robot_pose + camera_bearing) into a
2D probability grid over the arena map.  Each observation casts a
Gaussian-spread ray into the grid, weighted by detection confidence
and inversely by distance (closer = tighter bearing angle = more weight).

A spawn zone mask filters out impossible lifeboat positions:
  - Arena walls + 15 cm clearance
  - Pirate zone at arena Y midpoint (± 15 cm exclusion)
  - Opposite team's X half (arena split left / right for two teams)

Standalone — no ROS2 imports.  Run directly for a unit test:
    python3 lifeboat_triangulator.py

Usage from nav script:
    from lifeboat_triangulator import LifeboatTriangulator

    tri = LifeboatTriangulator(
        valid_x_min=-0.35, valid_x_max=0.15,   # left team
    )
    tri.add_observation(robot_x, robot_y, robot_yaw,
                        bearing_deg, confidence, ground_distance_cm)
    estimate = tri.get_best_estimate()
    if estimate:
        target_x, target_y = estimate

NOTE on bearing sign convention (distance_estimator.py):
  bearing_deg positive  = object is to the RIGHT of camera centre
  In ROS2 yaw,  positive yaw = counter-clockwise = robot's LEFT
  Therefore:   abs_bearing = robot_yaw - radians(bearing_deg)
  *** Verify this sign with a live test before competition. ***
"""

import math
import numpy as np


# ── Arena geometry (open waters SLAM frame) ────────────────────────
ARENA_X_MIN    = -0.5
ARENA_X_MAX    =  1.1
ARENA_Y_MIN    =  0.0
ARENA_Y_MAX    =  3.6
WALL_CLEARANCE =  0.15   # 6 inches — min clearance from walls / obstacles

# Pirate zone: ± 6 inches around arena Y midpoint (≈ 1.80 m)
ARENA_Y_MID    =  1.80
PIRATE_Y_MIN   =  ARENA_Y_MID - WALL_CLEARANCE   # 1.65
PIRATE_Y_MAX   =  ARENA_Y_MID + WALL_CLEARANCE   # 1.95

# Team X halves (arena X centre ≈ 0.30 m)
ARENA_X_MID    =  0.30
LEFT_X_MIN     =  ARENA_X_MIN + WALL_CLEARANCE   # -0.35
LEFT_X_MAX     =  ARENA_X_MID - WALL_CLEARANCE   #  0.15
RIGHT_X_MIN    =  ARENA_X_MID + WALL_CLEARANCE   #  0.45
RIGHT_X_MAX    =  ARENA_X_MAX - WALL_CLEARANCE   #  0.95

# Lifeboat Y range: both halves of arena (pirate zone splits them).
# Far half ends 6 inches + half port zone (0.30 m) south of target port (Y≈3.40).
TARGET_PORT_Y  =  3.40
PORT_HALF      =  0.30   # half of the 2 ft × 2 ft port tile
VALID_Y_MIN    =  ARENA_Y_MIN + WALL_CLEARANCE               #  0.15
VALID_Y_MAX    =  TARGET_PORT_Y - PORT_HALF - WALL_CLEARANCE #  2.95

# ── Ray-casting parameters ─────────────────────────────────────────
GRID_RESOLUTION  = 0.05    # 5 cm cells
MAX_RAY_RANGE    = 2.0     # m — beyond this bearing is too noisy
BASE_SIGMA_DEG   = 12.0    # angular half-width uncertainty (degrees)
CLOSE_RANGE_REF  = 0.30    # m — distance at which bearing is fully trusted
MIN_OBSERVATIONS = 2       # minimum rays before returning an estimate


class LifeboatTriangulator:
    """
    2D probability heatmap for lifeboat position estimation.

    Parameters
    ----------
    arena_x_min / x_max / y_min / y_max : float
        SLAM map boundaries (metres).
    grid_resolution : float
        Cell size in metres (default 0.05 = 5 cm).
    min_observations : int
        Minimum observations before get_best_estimate() returns a result.
    pirate_y_min / pirate_y_max : float
        Y range of pirate zone (excluded from spawn zone).
    valid_x_min / valid_x_max : float
        X range for this team's half.
    valid_y_max : float
        Maximum Y for lifeboat spawn (up to target port exclusion zone).
    wall_clearance : float
        Clearance from arena walls (metres).
    """

    def __init__(
        self,
        arena_x_min=ARENA_X_MIN, arena_x_max=ARENA_X_MAX,
        arena_y_min=ARENA_Y_MIN, arena_y_max=ARENA_Y_MAX,
        grid_resolution=GRID_RESOLUTION,
        min_observations=MIN_OBSERVATIONS,
        pirate_y_min=PIRATE_Y_MIN, pirate_y_max=PIRATE_Y_MAX,
        valid_x_min=LEFT_X_MIN, valid_x_max=LEFT_X_MAX,
        valid_y_max=VALID_Y_MAX,
        wall_clearance=WALL_CLEARANCE,
    ):
        self.x_min = arena_x_min
        self.x_max = arena_x_max
        self.y_min = arena_y_min
        self.y_max = arena_y_max
        self.res   = float(grid_resolution)
        self.min_observations = min_observations

        self.grid_w = int(math.ceil((arena_x_max - arena_x_min) / grid_resolution))
        self.grid_h = int(math.ceil((arena_y_max - arena_y_min) / grid_resolution))

        self._heatmap = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self._observation_count = 0

        self._spawn_mask = self._build_spawn_mask(
            pirate_y_min, pirate_y_max,
            valid_x_min, valid_x_max,
            valid_y_max, wall_clearance,
        )

    # ── Public API ────────────────────────────────────────────────

    @property
    def observation_count(self):
        return self._observation_count

    def add_observation(
        self,
        robot_x, robot_y, robot_yaw,
        bearing_deg, confidence, ground_distance_cm=100.0,
    ):
        """
        Record one bearing observation and add a weighted Gaussian beam
        into the heatmap.

        Parameters
        ----------
        robot_x, robot_y : float
            Robot map position (SLAM frame, metres).
        robot_yaw : float
            Robot heading (radians, ROS2 convention: positive = CCW).
        bearing_deg : float
            Camera bearing to lifeboat (degrees).
            Positive = right of camera centre, negative = left.
            (distance_estimator.py sign convention)
        confidence : float
            Detection confidence 0–1.
        ground_distance_cm : float
            Estimated ground distance in cm (from distance_estimator.py).
        """
        if confidence <= 0:
            return

        dist_m = max(0.10, min(MAX_RAY_RANGE, ground_distance_cm / 100.0))

        # Convert bearing to absolute map direction.
        # Camera positive bearing = object to robot's RIGHT = clockwise rotation
        # = negative yaw delta in ROS2 convention.
        # *** Verify sign against live test before competition. ***
        abs_bearing = robot_yaw - math.radians(bearing_deg)
        ray_dx = math.cos(abs_bearing)
        ray_dy = math.sin(abs_bearing)

        # Perpendicular direction for lateral beam spread
        perp_dx = -ray_dy
        perp_dy =  ray_dx

        # Angular uncertainty grows with distance; closer observations carry more weight
        distance_weight = min(1.0, CLOSE_RANGE_REF / dist_m)
        sigma_angular = math.radians(BASE_SIGMA_DEG) / max(0.1, distance_weight)

        # Along-ray Gaussian: concentrate evidence near estimated distance
        # sigma = 30% of estimated distance (minimum 10 cm)
        sigma_along = max(0.10, dist_m * 0.30)

        peak_weight = confidence * distance_weight
        step_m = self.res

        t = step_m
        while t <= MAX_RAY_RANGE:
            # Along-ray Gaussian weight
            along_w = math.exp(-0.5 * ((t - dist_m) / sigma_along) ** 2)
            w = peak_weight * along_w
            if w < 0.005:
                t += step_m
                continue

            # Lateral beam width in grid cells (grows with distance)
            sigma_lat = max(0.5, t * math.tan(sigma_angular) / self.res)

            cx = robot_x + t * ray_dx
            cy = robot_y + t * ray_dy

            # Spread across ±3 perpendicular cells
            for k in range(-3, 4):
                px = cx + k * self.res * perp_dx
                py = cy + k * self.res * perp_dy
                pi, pj = self._map_to_grid(px, py)
                if pi < 0 or pi >= self.grid_h or pj < 0 or pj >= self.grid_w:
                    continue
                if not self._spawn_mask[pi, pj]:
                    continue
                lat_w = math.exp(-0.5 * (k / sigma_lat) ** 2)
                self._heatmap[pi, pj] += w * lat_w

            t += step_m

        self._observation_count += 1

    def get_best_estimate(self):
        """
        Return (x, y) of the highest-confidence cell within the spawn zone,
        or None if fewer than min_observations have been recorded.
        """
        if self._observation_count < self.min_observations:
            return None

        masked = np.where(self._spawn_mask, self._heatmap, 0.0)
        if masked.max() <= 0:
            return None

        gi, gj = np.unravel_index(np.argmax(masked), masked.shape)
        x = self.x_min + (gj + 0.5) * self.res
        y = self.y_min + (gi + 0.5) * self.res
        return (float(x), float(y))

    def get_heatmap_copy(self):
        """Return a copy of the raw heatmap (for debugging / visualisation)."""
        return self._heatmap.copy()

    def reset(self):
        """Clear all accumulated observations."""
        self._heatmap[:] = 0.0
        self._observation_count = 0

    # ── Internal helpers ──────────────────────────────────────────

    def _map_to_grid(self, mx, my):
        """Convert map coordinates to (row, col) grid indices."""
        col = int((mx - self.x_min) / self.res)
        row = int((my - self.y_min) / self.res)
        return row, col

    def _build_spawn_mask(
        self,
        pirate_y_min, pirate_y_max,
        valid_x_min, valid_x_max,
        valid_y_max, wall_clearance,
    ):
        """Build boolean mask: True = valid spawn position."""
        mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        y_low  = self.y_min + wall_clearance
        y_high = min(valid_y_max, self.y_max - wall_clearance)

        for gi in range(self.grid_h):
            cy = self.y_min + (gi + 0.5) * self.res
            if cy < y_low or cy > y_high:
                continue
            if pirate_y_min <= cy <= pirate_y_max:
                continue
            for gj in range(self.grid_w):
                cx = self.x_min + (gj + 0.5) * self.res
                if cx < valid_x_min or cx > valid_x_max:
                    continue
                mask[gi, gj] = True

        return mask


# ─────────────────────────────────────────────────────────────────
# Unit test — run directly: python3 lifeboat_triangulator.py
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    print('=' * 60)
    print('LifeboatTriangulator unit test')
    print('=' * 60)

    PASS_THRESHOLD_M = 0.20   # 20 cm convergence target

    # ── Test 1: Left team, lifeboat at (-0.10, 0.80) ───────────────
    print('\nTest 1: Left team  — target (-0.10, 0.80)')
    tri = LifeboatTriangulator(valid_x_min=LEFT_X_MIN, valid_x_max=LEFT_X_MAX)

    TRUE_X, TRUE_Y = -0.10, 0.80
    observations = [
        # (robot_x, robot_y, robot_yaw, bearing_deg, confidence, dist_cm)
        (-0.05, 3.40, -math.pi/2,  3.0, 0.75, 260.0),
        (-0.05, 2.50, -math.pi/2, -2.0, 0.80, 170.0),
        ( 0.10, 1.75, -math.pi/2, -9.0, 0.85,  96.0),
        (-0.25, 1.75, -math.pi/2,  7.5, 0.70, 116.0),
        (-0.05, 1.10, -math.pi/2,  0.5, 0.90,  30.0),
    ]
    for i, args in enumerate(observations):
        tri.add_observation(*args)
    est = tri.get_best_estimate()
    if est:
        err = math.hypot(est[0] - TRUE_X, est[1] - TRUE_Y)
        status = 'PASS' if err < PASS_THRESHOLD_M else 'FAIL'
        print(f'  Estimate: ({est[0]:.3f}, {est[1]:.3f})  '
              f'true: ({TRUE_X:.3f}, {TRUE_Y:.3f})  '
              f'error: {err*100:.1f} cm  [{status}]')
    else:
        print('  FAIL — no estimate returned')

    # ── Test 2: Right team, lifeboat at (0.60, 1.20) ───────────────
    print('\nTest 2: Right team — target (0.60, 1.20)')
    tri2 = LifeboatTriangulator(valid_x_min=RIGHT_X_MIN, valid_x_max=RIGHT_X_MAX)
    TRUE_X2, TRUE_Y2 = 0.60, 1.20
    obs2 = [
        (0.80, 3.40, -math.pi/2, -8.0, 0.70, 226.0),
        (0.80, 2.50, -math.pi/2, -6.0, 0.80, 132.0),
        (0.60, 1.75, -math.pi/2,  0.0, 0.90,  55.0),
        (0.50, 1.75, -math.pi/2,  5.0, 0.80,  58.0),
    ]
    for args in obs2:
        tri2.add_observation(*args)
    est2 = tri2.get_best_estimate()
    if est2:
        err = math.hypot(est2[0] - TRUE_X2, est2[1] - TRUE_Y2)
        status = 'PASS' if err < PASS_THRESHOLD_M else 'FAIL'
        print(f'  Estimate: ({est2[0]:.3f}, {est2[1]:.3f})  '
              f'true: ({TRUE_X2:.3f}, {TRUE_Y2:.3f})  '
              f'error: {err*100:.1f} cm  [{status}]')
    else:
        print('  FAIL — no estimate returned')

    # ── Test 3: Insufficient observations ──────────────────────────
    print('\nTest 3: Insufficient observations (expect None)')
    tri3 = LifeboatTriangulator(min_observations=3)
    tri3.add_observation(0.0, 1.0, -math.pi/2, 0.0, 0.8, 50.0)
    tri3.add_observation(0.0, 0.5, -math.pi/2, 0.0, 0.8, 30.0)
    est3 = tri3.get_best_estimate()
    status = 'PASS' if est3 is None else 'FAIL'
    print(f'  Result: {est3}  [{status}]')

    # ── Spawn zone stats ───────────────────────────────────────────
    total = tri.grid_h * tri.grid_w
    valid = int(tri._spawn_mask.sum())
    pct   = 100.0 * valid / total
    print(f'\nSpawn zone (left): {valid} / {total} cells valid ({pct:.1f}%)')
    print('Grid size: '
          f'{tri.grid_w} cols × {tri.grid_h} rows = {total} cells '
          f'(resolution {GRID_RESOLUTION*100:.0f} cm)')
    print('=' * 60)
