#!/usr/bin/env python3
import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
import numpy as np

# --- SERIAL SETUP ---
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

SERIAL_PORT = '/dev/ttyUSB0' 
SERIAL_BAUD = 9600
CMD_DROP = b'\x02'

def setup_serial():
    if not HAS_SERIAL:
        print("⚠️ PySerial not installed. Serial commands disabled.")
        return None
    try:
        s = serial.Serial(SERIAL_PORT, SERIAL_BAUD, bytesize=8, parity='N', stopbits=1, timeout=0.1, dsrdtr=False, rtscts=False)
        s.dtr = False
        print(f"🔌 Serial connected safely on {SERIAL_PORT}")
        return s
    except Exception as e: 
        print(f"⚠️ Serial connection failed on {SERIAL_PORT}: {e}")
        return None

def create_pose(nav, x, y, yaw_deg):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = nav.get_clock().now().to_msg()
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    
    yaw = np.radians(yaw_deg)
    pose.pose.orientation.z = np.sin(yaw / 2.0)
    pose.pose.orientation.w = np.cos(yaw / 2.0)
    return pose

def main():
    rclpy.init()
    
    node = rclpy.create_node('lane_selector')
    node.declare_parameter('lane', 'left_open')
    lane_choice = node.get_parameter('lane').value
    node.destroy_node()

    # --- UPDATED COORDINATES ---
    open_water_coords = {
        'left_open':  {
            'start_x': 1.523, 
            'start_y': 0.305, 
            'goal_x': 1.523,    # Drives perfectly straight
            'goal_y': 3.77, 
            'home_yaw': -90.0
        },
        'right_open': {
            'start_x': 2.742, 
            'start_y': 0.305, 
            'goal_x': 2.850,    # Drives on a slight diagonal to the right
            'goal_y': 3.77, 
            'home_yaw': -90.0   # Fixed: Was 90.0, causing the wall crash
        }
    }
    
    if lane_choice not in open_water_coords:
        lane_choice = 'left_open'

    data = open_water_coords[lane_choice]
    nav = BasicNavigator()
    serial_conn = setup_serial()

    print("⏳ Waiting for Nav2/AMCL to stabilize from launch file...")
    nav.waitUntilNav2Active(localizer='amcl')
    print("✅ System Ready!")

    # 5. Mission: Drive to Port
    print(f"🚀 Navigating to Port Cargo in {lane_choice} (Goal X: {data['goal_x']})...")
    goal_pose = create_pose(nav, data['goal_x'], data['goal_y'], 90.0)
    nav.goToPose(goal_pose)

    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback:
            dist = feedback.distance_remaining
            print(f'Distance to port: {dist:.2f} m', end='\r')
            
            # Increased preempt from 0.40 to 0.50 for a safer turning radius
            if dist < 0.50: 
                print("\n🎯 Port reached! Deploying cargo and executing fast turnaround...")
                if serial_conn:
                    try:
                        serial_conn.write(CMD_DROP)
                    except Exception as e:
                        print(f"❌ Serial write failed: {e}")
                break

    # 6. Return Home
    print(f"🏁 Heading home to yaw: {data['home_yaw']}°")
    home_pose = create_pose(nav, data['start_x'], data['start_y'], data['home_yaw'])
    nav.goToPose(home_pose)
        
    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback:
            print(f'Distance to home: {feedback.distance_remaining:.2f} m', end='\r')

    print("\n🎉 Successfully returned home!")
    
    if serial_conn:
        serial_conn.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
