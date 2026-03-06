#! /usr/bin/env python3
"""
ECED3901 Challenge Arena Waypoint Navigator
Navigates from any of the 4 deployment positions to the target port and back.

Usage:
  ros2 run eced3901 arena_nav_wp.py                          # default: left_coastal
  ros2 run eced3901 arena_nav_wp.py --ros-args -p start:=left_coastal
  ros2 run eced3901 arena_nav_wp.py --ros-args -p start:=left_open
  ros2 run eced3901 arena_nav_wp.py --ros-args -p start:=right_open
  ros2 run eced3901 arena_nav_wp.py --ros-args -p start:=right_coastal

Arena layout (14ft x 14ft = 4.267m x 4.267m):
  Left coastal:  x=0 to 4ft  (zigzag walls)
  Open waters:   x=4ft to 10ft (pirates in middle)
  Right coastal: x=10ft to 14ft (zigzag walls)
  
  Deploy zones:  y=1ft  (bottom)
  Target ports:  y=12-14ft (top, 2ft x 2ft green squares)

Coastal zigzag walls at y=4ft, y=7ft, y=10ft alternate left/right.
"""

import sys
import numpy as np
from copy import deepcopy

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy
from rclpy.node import Node

# Unit conversion
FT = 0.3048

# Headings
NORTH = 1.5708   # π/2
SOUTH = -1.5708  # -π/2

# ============================================================
# Start/port positions for each deployment option
# ============================================================
POSITIONS = {
    'left_coastal': {
        'start':  (3*FT, 1*FT, NORTH),    # x=3ft, y=1ft
        'port':   (3*FT, 13*FT),           # center of 2x2ft port zone
    },
    'left_open': {
        'start':  (5*FT, 1*FT, NORTH),    # x=5ft, y=1ft
        'port':   (5*FT, 13*FT),
    },
    'right_open': {
        'start':  (9*FT, 1*FT, NORTH),    # x=9ft, y=1ft  
        'port':   (9*FT, 13*FT),
    },
    'right_coastal': {
        'start':  (11*FT, 1*FT, NORTH),   # x=11ft, y=1ft
        'port':   (11*FT, 13*FT),
    },
}

# ============================================================
# Coastal path waypoints (zigzag navigation)
# Wall positions: y=4ft, y=7ft, y=10ft
# Left coastal: walls at y=4ft from LEFT, y=7ft from RIGHT, y=10ft from LEFT
# Right coastal: walls at y=4ft from RIGHT, y=7ft from LEFT, y=10ft from RIGHT
# ============================================================

# Gap geometry (walls are 0.15 m thick, path width ≈ 1.07 m, gap ≈ 0.307 m)
#
# Left coastal (x = 0 .. 4ft):  outer-inner edges 0.075 .. 1.144
#   Wall from left:  x = 0.075–0.837 → gap 0.837–1.144, centre ≈ 0.99
#   Wall from right: x = 0.382–1.144 → gap 0.075–0.382, centre ≈ 0.23
LEFT_COAST_R = 0.99   # gap centre near divider (pass walls from left)
LEFT_COAST_L = 0.23   # gap centre near outer wall (pass walls from right)
LEFT_COAST_C = 0.61   # path centre

# Right coastal (x = 10ft .. 14ft):  inner-outer edges 3.123 .. 4.192
#   Wall from right: x = 3.430–4.192 → gap 3.123–3.430, centre ≈ 3.28
#   Wall from left:  x = 3.123–3.885 → gap 3.885–4.192, centre ≈ 4.04
RIGHT_COAST_L = 3.28  # gap centre near divider (pass walls from right)
RIGHT_COAST_R = 4.04  # gap centre near outer wall (pass walls from left)
RIGHT_COAST_C = 3.66  # path centre


def get_coastal_waypoints_forward(side):
    """Generate forward (north) waypoints through a coastal zigzag."""
    if side == 'left':
        R, L, C = LEFT_COAST_R, LEFT_COAST_L, LEFT_COAST_C
    else:
        # Right coastal: walls are mirrored
        # Wall 1 (y=4ft) from RIGHT outer wall → pass LEFT near divider
        # Wall 2 (y=7ft) from LEFT divider   → pass RIGHT near outer wall
        # Wall 3 (y=10ft) from RIGHT outer wall → pass LEFT near divider
        R, L, C = RIGHT_COAST_R, RIGHT_COAST_L, RIGHT_COAST_C

    waypoints = [
        # Align to the gap side before wall 1 (y=4 ft)
        [R if side == 'left' else L, 2.0*FT, NORTH],
        [R if side == 'left' else L, 3.5*FT, NORTH],     # just before wall 1
        [R if side == 'left' else L, 5.0*FT, NORTH],     # past wall 1
        # Cross over for wall 2 (y=7 ft)
        [L if side == 'left' else R, 5.5*FT, NORTH],
        [L if side == 'left' else R, 6.5*FT, NORTH],     # just before wall 2
        [L if side == 'left' else R, 8.0*FT, NORTH],     # past wall 2
        # Cross over for wall 3 (y=10 ft)
        [R if side == 'left' else L, 8.5*FT, NORTH],
        [R if side == 'left' else L, 9.5*FT, NORTH],     # just before wall 3
        [R if side == 'left' else L, 11.0*FT, NORTH],    # past wall 3
        # Enter port zone
        [C, 12.5*FT, NORTH],
    ]
    return waypoints


def get_coastal_waypoints_return(side):
    """Generate return (south) waypoints through a coastal zigzag."""
    if side == 'left':
        R, L, C = LEFT_COAST_R, LEFT_COAST_L, LEFT_COAST_C
    else:
        R, L, C = RIGHT_COAST_L, RIGHT_COAST_R, RIGHT_COAST_C

    waypoints = [
        # Head south from port
        [C, 11.5*FT, SOUTH],
        # Approach wall 3 (y=10ft) from above
        [R if side == 'left' else L, 10.5*FT, SOUTH],
        [R if side == 'left' else L, 9.0*FT, SOUTH],     # past wall 3
        # Cross for wall 2 (y=7ft)
        [L if side == 'left' else R, 8.5*FT, SOUTH],
        [L if side == 'left' else R, 7.5*FT, SOUTH],     # approach wall 2
        [L if side == 'left' else R, 6.0*FT, SOUTH],     # past wall 2
        # Cross for wall 1 (y=4ft)
        [R if side == 'left' else L, 5.5*FT, SOUTH],
        [R if side == 'left' else L, 4.5*FT, SOUTH],     # approach wall 1
        [R if side == 'left' else L, 3.0*FT, SOUTH],     # past wall 1
        # Return to deploy zone
        [C, 1.5*FT, SOUTH],
    ]
    return waypoints


def get_open_water_waypoints_forward(start_x):
    """Generate forward waypoints through open waters (straight shot, no zigzag)."""
    return [
        [start_x, 3*FT, NORTH],
        [start_x, 5*FT, NORTH],
        # Navigate around pirate zone (y≈6-8ft): slight lateral offset
        [start_x, 7*FT, NORTH],
        [start_x, 9*FT, NORTH],
        [start_x, 11*FT, NORTH],
        [start_x, 12.5*FT, NORTH],
    ]


def get_open_water_waypoints_return(start_x):
    """Generate return waypoints through open waters."""
    return [
        [start_x, 11*FT, SOUTH],
        [start_x, 9*FT, SOUTH],
        [start_x, 7*FT, SOUTH],
        [start_x, 5*FT, SOUTH],
        [start_x, 3*FT, SOUTH],
        [start_x, 1.5*FT, SOUTH],
    ]


def get_quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion [x, y, z, w]."""
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


def make_pose(x, y, yaw, navigator):
    """Create a PoseStamped from x, y, yaw."""
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    q = get_quaternion_from_euler(0, 0, yaw)
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    return pose


def follow_waypoints(navigator, waypoints, label=""):
    """Send waypoints to Nav2 and monitor progress."""
    poses = [make_pose(wp[0], wp[1], wp[2], navigator) for wp in waypoints]
    navigator.followWaypoints(poses)
    
    i = 0
    while not navigator.isTaskComplete():
        i += 1
        feedback = navigator.getFeedback()
        if feedback and i % 5 == 0:
            print(f'  {label} waypoint: {feedback.current_waypoint + 1}/{len(poses)}')

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print(f'{label} navigation complete!')
        return True
    elif result == TaskResult.CANCELED:
        print(f'{label} navigation was canceled.')
        return False
    elif result == TaskResult.FAILED:
        print(f'{label} navigation failed!')
        return False
    return False


def main():
    rclpy.init()
    
    # Create a temporary node to read the 'start' parameter
    param_node = Node('arena_nav_params')
    param_node.declare_parameter('start', 'left_coastal')
    start_pos = param_node.get_parameter('start').get_parameter_value().string_value
    param_node.destroy_node()
    
    if start_pos not in POSITIONS:
        print(f"ERROR: Unknown start position '{start_pos}'")
        print(f"Valid options: {list(POSITIONS.keys())}")
        exit(1)
    
    pos = POSITIONS[start_pos]
    sx, sy, syaw = pos['start']
    px, py = pos['port']
    
    print(f"=== ECED3901 Challenge Arena Navigation ===")
    print(f"Start position: {start_pos}")
    print(f"  Deploy: ({sx:.3f}, {sy:.3f}) heading {'NORTH' if syaw > 0 else 'SOUTH'}")
    print(f"  Target port: ({px:.3f}, {py:.3f})")
    
    navigator = BasicNavigator()

    # Set initial pose
    initial_pose = make_pose(sx, sy, syaw, navigator)
    navigator.setInitialPose(initial_pose)

    # Wait for Nav2 — AMCL + bt_navigator are both lifecycle-managed nodes
    print('\nWaiting for Nav2 to activate...')
    navigator.waitUntilNav2Active()
    print('Nav2 active!')

    # ---- Build forward waypoints ----
    is_coastal = 'coastal' in start_pos
    
    if is_coastal:
        side = 'left' if 'left' in start_pos else 'right'
        forward_wps = get_coastal_waypoints_forward(side)
        # Add final waypoint at port center
        forward_wps.append([px, py, NORTH])
    else:
        forward_wps = get_open_water_waypoints_forward(sx)
        forward_wps.append([px, py, NORTH])

    # ---- Navigate forward ----
    print(f'\n>>> FORWARD: Heading to target port ({len(forward_wps)} waypoints)')
    if not follow_waypoints(navigator, forward_wps, "Forward"):
        print("Forward navigation failed. Exiting.")
        exit(1)

    print("\n--- Reached target port! ---")
    print("(Cargo drop-off / pickup would happen here)")

    # ---- Build return waypoints ----
    if is_coastal:
        return_wps = get_coastal_waypoints_return(side)
        return_wps.append([sx, sy, SOUTH])
    else:
        return_wps = get_open_water_waypoints_return(sx)
        return_wps.append([sx, sy, SOUTH])

    # ---- Navigate return ----
    print(f'\n>>> RETURN: Heading back to home port ({len(return_wps)} waypoints)')
    if not follow_waypoints(navigator, return_wps, "Return"):
        print("Return navigation failed.")
        exit(1)

    print("\n=== Challenge navigation complete! ===")
    exit(0)


if __name__ == '__main__':
    main()
