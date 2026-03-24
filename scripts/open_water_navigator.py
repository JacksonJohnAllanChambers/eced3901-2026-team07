#!/usr/bin/env python3
import rclpy
import numpy as np
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped

def create_pose(nav, x, y, yaw_deg):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = nav.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    
    # Convert degrees to quaternion for ROS
    yaw = np.radians(yaw_deg)
    pose.pose.orientation.z = np.sin(yaw / 2.0)
    pose.pose.orientation.w = np.cos(yaw / 2.0)
    return pose

def main():
    rclpy.init()
    nav = BasicNavigator()

    # --- 1. WAIT FOR NAV2 (SLAM BYPASS) ---
    print("Waiting for Nav2 to activate...")
    # This specifically tells the script to ignore SLAM and only check the behavior tree!
    nav.waitUntilNav2Active(navigator='bt_navigator', localizer='bt_navigator')
    print(" Nav2 is active! Generating SLAM map...")

    # --- 2. DEFINE THE STRAIGHT PATH (Open Water) ---
    # Because SLAM starts blind, the robot's starting position is ALWAYS (0.0, 0.0)
    # We are commanding it to drive straight up the Y-axis by 3.5 meters.
    path = [
        create_pose(nav, 0.0, 1.5, 90.0),   # Mid-way point (1.5 meters forward)
        create_pose(nav, 0.0, 3.5, 90.0)    # Destination Port (3.5 meters forward)
    ]

    print(" Starting Open Water run...")
    nav.followWaypoints(path)

    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback:
            print(f'Navigating... Distance remaining to next waypoint: {feedback.distance_remaining:.2f} m')

    # --- 3. RETURN TO START ---
    print(" Destination reached! Turning around to return to (0,0)...")
    
    # Turn around and face "down" (-90 degrees) to go back to exactly where it spawned
    return_pose = create_pose(nav, 0.0, 0.0, -90.0)
    nav.goToPose(return_pose)

    while not nav.isTaskComplete():
        pass

    print(" Back at deploy zone. Mission complete.")
    rclpy.shutdown()

if __name__ == '__main__':
    main()
