#!/usr/bin/env python3
import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
import numpy as np

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
    
    # 1. Create a temporary node to read the 'lane' parameter
    node = rclpy.create_node('lane_selector')
    node.declare_parameter('lane', 'left_open')
    lane_choice = node.get_parameter('lane').value
    node.destroy_node()

    # 2. Open Water Coordinate Map (Based on 2ft grid)
    # Left Open = 5 feet (1.524m), Right Open = 9 feet (2.743m)
   open_water_coords = {
        'left_open':  {
            'x': 1.524, 
            'y': 0.305, 
            'goal_y': 3.35, 
            'home_yaw': -90.0  # Turn clockwise
        },
        'right_open': {
            'x': 2.743, 
            'y': 0.305, 
            'goal_y': 3.35, 
            'home_yaw': 90.0   # Turn counter-clockwise (away from center wall)
        }
    }

    if lane_choice not in open_water_coords:
        print(f"⚠️  Warning: '{lane_choice}' is not an Open Water lane. Defaulting to left_open.")
        lane_choice = 'left_open'

    data = open_water_coords[lane_choice]
    nav = BasicNavigator()

    # 3. Teleport AMCL to the chosen lane
    print(f"📍 Initializing AMCL position for {lane_choice.upper()}...")
    init_pose = create_pose(nav, data['x'], data['y'], 90.0)
    nav.setInitialPose(init_pose)

    # 4. Wait for Nav2 to wake up
    print("⏳ Waiting for Nav2/AMCL to stabilize...")
    nav.waitUntilNav2Active(localizer='amcl')
    print("✅ System Ready!")

    # 5. Mission: Drive to Port
    print(f"🚀 Navigating to Port Cargo in {lane_choice}...")
    goal_pose = create_pose(nav, data['x'], data['goal_y'], 90.0)
    nav.goToPose(goal_pose)

    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback:
            print(f'Distance to port: {feedback.distance_remaining:.2f} m', end='\r')

    # 6. Check Result and Return Home
   if nav.getResult() == TaskResult.SUCCEEDED:
        print(f"\n🏁 Reached Port! Turning to {data['home_yaw']}° and heading home...")
        
        # We use the dynamic 'home_yaw' here
        home_pose = create_pose(nav, data['x'], data['y'], data['home_yaw'])
        nav.goToPose(home_pose)
        
        while not nav.isTaskComplete():
            pass
        print("🎉 Successfully returned home!")
    else:
        print("\n❌ Mission Aborted. Please check for obstacles or AMCL drift.")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
