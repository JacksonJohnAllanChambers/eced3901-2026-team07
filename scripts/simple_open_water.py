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
    
    yaw = np.radians(yaw_deg)
    pose.pose.orientation.z = np.sin(yaw / 2.0)
    pose.pose.orientation.w = np.cos(yaw / 2.0)
    return pose

def main():
    rclpy.init()
    nav = BasicNavigator()

    # Left Open lane perfect starting coordinates
    start_x = 1.524
    start_y = 0.305
    port_y = 3.35

    print("📍 Injecting mathematically perfect starting pose into AMCL...")
    initial_pose = create_pose(nav, start_x, start_y, 90.0)
    nav.setInitialPose(initial_pose)
    
    print("⏳ Waiting for Nav2 and AMCL to wake up...")
    # This ensures we don't send commands before the map is ready
    nav.waitUntilNav2Active(localizer='amcl')
    print("✅ AMCL is active! Ready to navigate.")

    
    # --- GOAL 1: DRIVE OUT ---
    print("🚀 Driving straight up the Open Water lane...")
    target_pose = create_pose(nav, start_x, port_y, 90.0)
    nav.goToPose(target_pose)

    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback:
            print(f'Distance to port: {feedback.distance_remaining:.2f} m', end='\r')
# --- THE SAFETY CHECK ---
    result = nav.getResult()
    if result != TaskResult.SUCCEEDED:
        print("\n❌ Nav2 aborted the mission! Stopping robot to prevent crash.")
        rclpy.shutdown()
        return  # Exit the script entirely!

    # --- GOAL 2: COME HOME ---
    print("\n🏁 Reached the port! Turning around and heading home...")
    home_pose = create_pose(nav, start_x, start_y, -90.0)
    nav.goToPose(home_pose)

    while not nav.isTaskComplete():
        pass

if __name__ == '__main__':
    main()
