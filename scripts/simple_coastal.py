#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
import math
import sys
import time

def create_pose(navigator, x, y, yaw_deg):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = navigator.get_clock().now().to_msg()
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    yaw_rad = math.radians(yaw_deg)
    pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
    pose.pose.orientation.w = math.cos(yaw_rad / 2.0)
    return pose

def main():
    rclpy.init()
    node = Node('simple_coastal_navigator')
    node.declare_parameter('lane', 'left_coastal')
    lane = node.get_parameter('lane').value
    
    navigator = BasicNavigator()

    if lane == 'left_coastal':
        start_x = 0.914
        outbound_coords = [
            (0.960, 0.90, 90.0),  # Align early
            (0.960, 1.15, 90.0),  # Pre-Gap 1 
            (0.960, 1.50, 90.0),  # Post-Gap 1 
            (0.375, 1.75, 90.0),  # Pre-Gap 2 (Shift LEFT)
            (0.355, 2.29, 90.0),  # Post-Gap 2 
            (0.860, 2.49, 90.0),  # Pre-Gap 3 (Shift RIGHT)
            (0.860, 3.30, 90.0),  # Post-Gap 4 (Straightaway)
            (0.914, 3.67, 90.0)   # Final straight shot to Port Cargo
        ]
        
        # EXPLICIT RETURN PATH
        return_coords = [
            (0.914, 3.55, -90.0), # SAFE TURNAROUND: Spin 180 in the open port area
            (0.860, 3.30, -90.0), # Align with Gaps 4 & 3
            (0.860, 2.58, -90.0), # Drive straight through Gaps 4 & 3
            (0.400, 2.40, -90.0), # Shift LEFT for Gap 2
            (0.395, 1.75, -90.0), # Drive straight through Gap 2
            (0.960, 1.55, -90.0), # Shift RIGHT for Gap 1
            (0.960, 0.90, -90.0)  # Drive straight through Gap 1
        ]
        
    elif lane == 'right_coastal':
        start_x = 3.353
        outbound_coords = [
            (3.300, 0.90, 90.0),  
            (3.300, 1.52, 90.0),  
            (3.410, 1.52, 90.0),  
            (3.410, 2.13, 90.0),  
            (3.300, 2.13, 90.0),  
            (3.300, 2.74, 90.0),  
            (3.410, 2.74, 90.0),  
            (3.410, 3.36, 90.0),  
            (3.353, 3.36, 90.0),  
            (3.353, 3.67, 90.0)   
        ]
        
        # EXPLICIT RETURN PATH
        return_coords = [
            (3.353, 3.55, -90.0), # SAFE TURNAROUND: Spin 180 in the open port area
            (3.353, 3.36, -90.0), 
            (3.410, 3.36, -90.0), 
            (3.410, 2.74, -90.0), 
            (3.300, 2.74, -90.0), 
            (3.300, 2.13, -90.0), 
            (3.410, 2.13, -90.0), 
            (3.410, 1.52, -90.0), 
            (3.300, 1.52, -90.0), 
            (3.300, 0.90, -90.0)  
        ]
        
    else:
        print("Invalid lane! Use 'left_coastal' or 'right_coastal'")
        sys.exit(1)

    print(f"\n📍 Initializing for {lane.upper()}...")
    initial_pose = create_pose(navigator, start_x, 0.305, 90.0)
    navigator.setInitialPose(initial_pose)
    navigator.waitUntilNav2Active()
    print("✅ System Ready!\n")

    print("🚀 Starting Outbound Slalom...")
    for i, (x, y, yaw) in enumerate(outbound_coords):
        target_name = f"Waypoint {i+1}" if i < len(outbound_coords)-1 else "Port Cargo"
        print(f"   -> Navigating to {target_name} at (X: {x}, Y: {y})")
        
        navigator.goToPose(create_pose(navigator, x, y, yaw))
        
        while not navigator.isTaskComplete():
            time.sleep(0.1) 
            
        if navigator.getResult() != TaskResult.SUCCEEDED:
            print(f"❌ Failed to reach {target_name}! Halting.")
            sys.exit(1)
            
    print("🏁 Reached Port! Turning around and heading home...\n")

    print("🚀 Starting Return Slalom...")
    
    for i, (x, y, yaw) in enumerate(return_coords):
        target_name = f"Return Waypoint {i+1}"
        print(f"   -> Navigating back through {target_name} at (X: {x}, Y: {y})")
        
        navigator.goToPose(create_pose(navigator, x, y, yaw))
        
        while not navigator.isTaskComplete():
            time.sleep(0.1)
            
        if navigator.getResult() != TaskResult.SUCCEEDED:
            print(f"❌ Failed to reach {target_name} on the way back! Halting.")
            sys.exit(1)

    print(f"   -> Navigating to Home at (X: {start_x}, Y: 0.305)")
    navigator.goToPose(create_pose(navigator, start_x, 0.305, -90.0))
    
    while not navigator.isTaskComplete():
        time.sleep(0.1)

    if navigator.getResult() == TaskResult.SUCCEEDED:
        print("🎉 Successfully navigated the coastal grid and returned home!")
    else:
        print("❌ Failed final approach to home!")

    navigator.lifecycleShutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
