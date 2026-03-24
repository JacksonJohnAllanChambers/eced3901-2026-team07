#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
import time
import serial

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

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

def send_cargo_cmd(ser, cmd_bytes, label):
    if ser is None:
        print(f"⚠️  Serial disconnected. (Simulating {label} command)")
        return
    try:
        ser.write(cmd_bytes)
        ser.flush()
        print(f"⚡ UART: Sent '{label}' command successfully.")
    except Exception as e:
        print(f"❌ UART Error sending {label}: {e}")

def blind_drive(pub, speed, duration):
    """Bypasses Nav2 to forcefully drive the wheels for a set time."""
    cmd = Twist()
    cmd.linear.x = float(speed)
    
    t0 = time.time()
    while time.time() - t0 < duration:
        pub.publish(cmd)
        time.sleep(0.05)
        
    # Send stop command
    pub.publish(Twist())

# ═══════════════════════════════════════════════════════════════
# MAIN SEQUENCE
# ═══════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    
    # 1. Setup Parameters & Simple Publisher
    setup_node = rclpy.create_node('mission_setup')
    setup_node.declare_parameter('lane', 'left_open')
    lane_choice = setup_node.get_parameter('lane').value
    
    # We need a raw cmd_vel publisher for our Blind Nudges
    cmd_pub = setup_node.create_publisher(Twist, 'cmd_vel', 10)

    # 2. Setup UART
    try:
        cargo_serial = serial.Serial('/dev/ttyUSB4', 9600, timeout=1.0)
        print("🔌 Cargo ATmega connected on /dev/ttyUSB4")
    except Exception as e:
        cargo_serial = None
        print(f"⚠️  Warning: Cargo serial FAILED to open: {e}")

    open_water_coords = {
        'left_open':  {
            'x': 1.523, 'y': 0.305, 'goal_y': 3.67, 
            'spin_to_cargo': -1.5708, # Spin Right (-90)
            'spin_to_home': -1.5708   # Spin Right to face South
        },
        'right_open': {
            'x': 2.742, 'y': 0.305, 'goal_y': 3.67, 
            'spin_to_cargo': 1.5708,  # Spin Left (+90)
            'spin_to_home': 1.5708    # Spin Left to face South
        }
    }
    
    if lane_choice not in open_water_coords: 
        lane_choice = 'left_open'
    data = open_water_coords[lane_choice]
    
    nav = BasicNavigator()

    print(f"📍 Initializing AMCL position for {lane_choice.upper()}...")
    init_pose = create_pose(nav, data['x'], data['y'], 90.0)
    nav.setInitialPose(init_pose)
    nav.waitUntilNav2Active(localizer='amcl')
    print("✅ System Ready!")

    # ── PHASE 1: Drive to Port (Nav2) ──
    print(f"🚀 Navigating to Port in {lane_choice}...")
    goal_pose = create_pose(nav, data['x'], data['goal_y'], 90.0)
    nav.goToPose(goal_pose)

    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback:
            dist = feedback.distance_remaining
            print(f'Distance to port wall: {dist:.2f} m', end='\r')
            if dist < 0.15: 
                print("\n🎯 Drop-off zone reached! Halting Nav2...")
                nav.cancelTask()
                break

    # ── PHASE 2: Drop Cargo (UART) ──
    print("📦 Deploying Cargo...")
    send_cargo_cmd(cargo_serial, b'2', 'DROP')
    time.sleep(2.0)  

    # ── PHASE 3: Spin to face New Cargo (Nav2) ──
    print("🔄 Spinning 90 degrees to face new payload...")
    nav.spin(spin_dist=data['spin_to_cargo'], time_allowance=10)
    while not nav.isTaskComplete(): pass

    # ── PHASE 4: Approach Cargo Bubble (Nav2) ──
    print("🚀 Driving to Cargo safety perimeter...")
    pickup_pose = create_pose(nav, data['x'], data['goal_y'], data['spin_to_cargo'])
    nav.goToPose(pickup_pose)

    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback:
            dist = feedback.distance_remaining
            print(f'Distance to cargo: {dist:.2f} m', end='\r')
            # Stop 15cm short so Nav2 doesn't abort due to Costmap collision
            if dist < 0.15: 
                print("\n🎯 At cargo perimeter. Halting Nav2...")
                nav.cancelTask()
                break

    # ── PHASE 5: The "Blind Nudge" (Raw Motor Command) ──
    print("🚗 Blind Nudging forward to make physical contact...")
    # Drive forward at 0.08 m/s for 2.0 seconds
    blind_drive(cmd_pub, speed=0.08, duration=2.0)

    # ── PHASE 6: Pickup Cargo (UART) ──
    print("🧲 Activating pickup mechanism...")
    send_cargo_cmd(cargo_serial, b'1', 'PICKUP')
    time.sleep(2.0)

    # ── PHASE 7: Reverse Out (Raw Motor Command) ──
    print("⏪ Reversing out of cargo pocket to clear costmap...")
    # Drive backward at -0.10 m/s for 2.0 seconds
    blind_drive(cmd_pub, speed=-0.10, duration=2.0)

    # ── PHASE 8: Spin South & Return Home (Nav2) ──
    print("🔄 Spinning to face South...")
    nav.spin(spin_dist=data['spin_to_home'], time_allowance=10)
    while not nav.isTaskComplete(): pass

    print(f"🏁 Heading home...")
    home_pose = create_pose(nav, data['x'], data['y'], -90.0)
    nav.goToPose(home_pose)
        
    while not nav.isTaskComplete(): pass

    print("\n🎉 Mission Complete! Cargo exchanged successfully.")
    
    if cargo_serial is not None:
        cargo_serial.close()
    
    setup_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
