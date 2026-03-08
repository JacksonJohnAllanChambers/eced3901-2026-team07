#!/usr/bin/env python3
"""
Generate a static occupancy grid map (.pgm + .yaml) of the ECED3901 challenge
arena from the known wall coordinates.

This map is used with AMCL for localization instead of building one from
scratch with SLAM.  Because we know the exact arena layout, the robot can
navigate immediately with full global context.

Run once:
    python3 scripts/generate_arena_map.py

Outputs:
    maps/arena_map.pgm
    maps/arena_map.yaml
"""

import numpy as np
import os

# ── Arena constants (must match challenge_arena.world) ────────────────
ARENA = 4.267          # 14 ft in metres
WALL_T = 0.15          # wall thickness (metres)
FT = 0.3048

# ── Map parameters ────────────────────────────────────────────────────
RES = 0.02             # metres per pixel (finer than costmap's 0.05)
MARGIN = 0.25          # extra margin around the arena
MAP_M = ARENA + 2 * MARGIN
MAP_PX = int(np.ceil(MAP_M / RES))

ORIGIN_X = -MARGIN
ORIGIN_Y = -MARGIN


def fill_wall(grid, cx, cy, sx, sy):
    """Mark a rectangular wall as occupied (0) on *grid*.

    cx, cy – centre of the wall in world metres
    sx, sy – full size of the wall in world metres
    """
    x0 = cx - sx / 2
    x1 = cx + sx / 2
    y0 = cy - sy / 2
    y1 = cy + sy / 2

    # Convert to pixel indices (row = y, col = x)
    c0 = max(0, int((x0 - ORIGIN_X) / RES))
    c1 = min(MAP_PX - 1, int(np.ceil((x1 - ORIGIN_X) / RES)))
    r0 = max(0, int((y0 - ORIGIN_Y) / RES))
    r1 = min(MAP_PX - 1, int(np.ceil((y1 - ORIGIN_Y) / RES)))

    grid[r0:r1 + 1, c0:c1 + 1] = 0          # occupied


def main():
    # Start with a free map (254 = free-space in ROS convention)
    grid = np.full((MAP_PX, MAP_PX), 254, dtype=np.uint8)

    # ── Perimeter walls ───────────────────────────────────────────────
    fill_wall(grid, ARENA / 2, 0,         ARENA, WALL_T)   # bottom
    fill_wall(grid, ARENA / 2, ARENA,     ARENA, WALL_T)   # top
    fill_wall(grid, 0,         ARENA / 2, WALL_T, ARENA)   # left
    fill_wall(grid, ARENA,     ARENA / 2, WALL_T, ARENA)   # right

    # ── Lane dividers (full-height) ──────────────────────────────────
    fill_wall(grid, 4 * FT,  ARENA / 2, WALL_T, ARENA)     # left  (x = 4 ft)
    fill_wall(grid, 10 * FT, ARENA / 2, WALL_T, ARENA)     # right (x = 10 ft)

    # ── Left coastal zigzag walls ────────────────────────────────────
    fill_wall(grid, 0.456, 4 * FT,  0.762, WALL_T)         # wall 1 from left
    fill_wall(grid, 0.763, 7 * FT,  0.762, WALL_T)         # wall 2 from right
    fill_wall(grid, 0.456, 10 * FT, 0.762, WALL_T)         # wall 3 from left

    # ── Right coastal zigzag walls ───────────────────────────────────
    fill_wall(grid, 3.811, 4 * FT,  0.762, WALL_T)         # wall 1 from right
    fill_wall(grid, 3.504, 7 * FT,  0.762, WALL_T)         # wall 2 from left
    fill_wall(grid, 3.811, 10 * FT, 0.762, WALL_T)         # wall 3 from right

    # ── Mark outside arena as unknown (205) ──────────────────────────
    for r in range(MAP_PX):
        wy = ORIGIN_Y + r * RES
        for c in range(MAP_PX):
            wx = ORIGIN_X + c * RES
            if (wx < -WALL_T / 2 or wx > ARENA + WALL_T / 2 or
                    wy < -WALL_T / 2 or wy > ARENA + WALL_T / 2):
                grid[r, c] = 205              # unknown

    # ── Write PGM (row 0 = top of image → flip y for ROS) ───────────
    grid_pgm = np.flipud(grid)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    maps_dir = os.path.join(script_dir, '..', 'maps')
    os.makedirs(maps_dir, exist_ok=True)

    pgm_path = os.path.join(maps_dir, 'arena_map.pgm')
    yaml_path = os.path.join(maps_dir, 'arena_map.yaml')

    with open(pgm_path, 'wb') as f:
        header = f"P5\n{MAP_PX} {MAP_PX}\n255\n"
        f.write(header.encode())
        f.write(grid_pgm.tobytes())

    with open(yaml_path, 'w') as f:
        f.write(f"image: arena_map.pgm\n")
        f.write(f"resolution: {RES}\n")
        f.write(f"origin: [{ORIGIN_X}, {ORIGIN_Y}, 0.0]\n")
        f.write(f"negate: 0\n")
        f.write(f"occupied_thresh: 0.65\n")
        f.write(f"free_thresh: 0.196\n")
        f.write(f"mode: trinary\n")

    print(f"Arena map generated: {MAP_PX}×{MAP_PX} px, "
          f"resolution = {RES} m/px")
    print(f"  PGM : {os.path.abspath(pgm_path)}")
    print(f"  YAML: {os.path.abspath(yaml_path)}")


if __name__ == '__main__':
    main()
