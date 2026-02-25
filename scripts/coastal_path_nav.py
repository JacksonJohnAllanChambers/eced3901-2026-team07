#! /usr/bin/env python3
# Coastal Path Waypoint Navigation
# Navigates through the coastal path corridor using Nav2, then returns to start.
# 
# The coastal path is a ~1.2m wide corridor running north-south (y: -2.1 to 2.0)
# centered around x ≈ -1.2, with 3 internal walls creating a zigzag pattern:
#   Wall_10 (y≈-1.0): protrudes from LEFT  -> pass on RIGHT
#   Wall_12 (y≈-0.2): protrudes from RIGHT -> pass on LEFT  
#   Wall_15 (y≈ 0.9): protrudes from LEFT  -> pass on RIGHT
#
# Robot spawns at (-1.2, -1.8) facing north (yaw=π/2)

import numpy as np
from copy import deepcopy

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import rclpy


def get_quaternion_from_euler(roll, pitch, yaw):
    """Convert Euler angles to quaternion [x, y, z, w]."""
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]


def main():
    rclpy.init()

    navigator = BasicNavigator()

    # --- Corridor geometry ---
    # Left wall inner edge:  x ≈ -1.73
    # Right wall inner edge: x ≈ -0.68
    # Corridor center:       x ≈ -1.20
    RIGHT_SIDE = -0.85   # x coord to pass walls protruding from left
    LEFT_SIDE  = -1.55   # x coord to pass walls protruding from right
    CENTER     = -1.20   # corridor center x
    
    HEADING_NORTH = 1.5708   # π/2
    HEADING_SOUTH = -1.5708  # -π/2

    # Waypoints: [x, y, yaw] — navigate the zigzag going north
    forward_route = [
        # Start area, head toward the right to pass Wall_10
        [RIGHT_SIDE, -1.4, HEADING_NORTH],
        # Past Wall_10 (y≈-1.02), stay right then cut left for Wall_12
        [RIGHT_SIDE, -0.6, HEADING_NORTH],
        # Go to left side to pass Wall_12 (y≈-0.16, protrudes from right)
        [LEFT_SIDE,  -0.2, HEADING_NORTH],
        # Past Wall_12, head back right for Wall_15
        [LEFT_SIDE,   0.4, HEADING_NORTH],
        # Go to right side to pass Wall_15 (y≈0.89, protrudes from left)
        [RIGHT_SIDE,  0.6, HEADING_NORTH],
        # Past Wall_15, continue north to end
        [RIGHT_SIDE,  1.2, HEADING_NORTH],
        # End of corridor
        [CENTER,      1.7, HEADING_NORTH],
    ]

    # Return route — reverse the zigzag going south
    return_route = [
        [RIGHT_SIDE,  1.2, HEADING_SOUTH],
        # Approach Wall_15 from above, pass on right
        [RIGHT_SIDE,  0.6, HEADING_SOUTH],
        # Cut to left side
        [LEFT_SIDE,   0.4, HEADING_SOUTH],
        # Past Wall_12 going south, pass on left
        [LEFT_SIDE,  -0.2, HEADING_SOUTH],
        # Cut to right side
        [RIGHT_SIDE, -0.6, HEADING_SOUTH],
        # Past Wall_10, continue south
        [RIGHT_SIDE, -1.4, HEADING_SOUTH],
        # Back to start
        [CENTER,     -1.7, HEADING_SOUTH],
    ]

    # Set initial pose (matching the world file spawn)
    initial_pose = PoseStamped()
    initial_pose.header.frame_id = 'map'
    initial_pose.header.stamp = navigator.get_clock().now().to_msg()
    initial_pose.pose.position.x = -1.2
    initial_pose.pose.position.y = -1.8
    q = get_quaternion_from_euler(0, 0, HEADING_NORTH)
    initial_pose.pose.orientation.x = q[0]
    initial_pose.pose.orientation.y = q[1]
    initial_pose.pose.orientation.z = q[2]
    initial_pose.pose.orientation.w = q[3]
    navigator.setInitialPose(initial_pose)

    # Wait for navigation to fully activate
    # slam_toolbox sync node auto-activates (no lifecycle get_state service),
    # so we skip the localizer wait and only wait for bt_navigator.
    import time
    print('Waiting for Nav2 to activate...')
    print('  (giving slam_toolbox 5 s to initialise)')
    time.sleep(5)
    navigator._waitForNodeToActivate('bt_navigator')
    print('Nav2 active! Starting coastal path navigation.')

    # --- Navigate forward through the corridor ---
    print('\n=== FORWARD: Navigating to the end of the coastal path ===')
    forward_points = []
    wp = PoseStamped()
    wp.header.frame_id = 'map'
    wp.header.stamp = navigator.get_clock().now().to_msg()
    for pt in forward_route:
        wp.pose.position.x = pt[0]
        wp.pose.position.y = pt[1]
        q = get_quaternion_from_euler(0, 0, pt[2])
        wp.pose.orientation.x = q[0]
        wp.pose.orientation.y = q[1]
        wp.pose.orientation.z = q[2]
        wp.pose.orientation.w = q[3]
        forward_points.append(deepcopy(wp))

    navigator.followWaypoints(forward_points)

    i = 0
    while not navigator.isTaskComplete():
        i += 1
        feedback = navigator.getFeedback()
        if feedback and i % 5 == 0:
            print(f'  Forward waypoint: {feedback.current_waypoint + 1}/{len(forward_points)}')

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print('Reached the end of the coastal path!')
    elif result == TaskResult.CANCELED:
        print('Forward navigation was canceled.')
        exit(1)
    elif result == TaskResult.FAILED:
        print('Forward navigation failed!')
        exit(1)

    # --- Navigate back through the corridor ---
    print('\n=== RETURN: Navigating back to the start ===')
    return_points = []
    wp.header.stamp = navigator.get_clock().now().to_msg()
    for pt in return_route:
        wp.pose.position.x = pt[0]
        wp.pose.position.y = pt[1]
        q = get_quaternion_from_euler(0, 0, pt[2])
        wp.pose.orientation.x = q[0]
        wp.pose.orientation.y = q[1]
        wp.pose.orientation.z = q[2]
        wp.pose.orientation.w = q[3]
        return_points.append(deepcopy(wp))

    navigator.followWaypoints(return_points)

    i = 0
    while not navigator.isTaskComplete():
        i += 1
        feedback = navigator.getFeedback()
        if feedback and i % 5 == 0:
            print(f'  Return waypoint: {feedback.current_waypoint + 1}/{len(return_points)}')

    result = navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        print('\nCoastal path round trip complete!')
    elif result == TaskResult.CANCELED:
        print('\nReturn navigation was canceled.')
    elif result == TaskResult.FAILED:
        print('\nReturn navigation failed!')

    exit(0)


if __name__ == '__main__':
    main()
