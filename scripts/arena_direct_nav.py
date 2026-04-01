#!/usr/bin/env python3
"""
pure_dead_reckoning.py — Open-Loop Autonomous Drop-Off
======================================================
No AMCL, No Nav2, No LIDAR. Pure timed velocity commands.
"""

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# --- SERIAL SETUP ---
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

SERIAL_PORT = '/dev/ttyUSB0' # Verify this!
SERIAL_BAUD = 9600
CMD_DROP = b'\x02'

# ═══════════════════════════════════════════════════════════════
#  TUNING PARAMETERS (Adjust these based on physical testing!)
# ═══════════════════════════════════════════════════════════════
# Distance needed: ~3.36 meters (Goal 3.67m - Start 0.305m)
CRUISE_SPEED = 0.15        # m/s
DRIVE_FWD_TIME = 22.4      # seconds (0.15 m/s * 22.4s = 3.36m)
DRIVE_REV_TIME = 3.0       # seconds backing away from port
TURN_SPEED = 0.5           # rad/s
TURN_180_TIME = 6.28       # seconds (math.pi / 0.5)
DRIVE_HOME_TIME = 20.0     # seconds (Slightly less than FWD to account for reverse)

class DeadReckoner(Node):
    def __init__(self):
        super().__init__('dead_reckoning_mission')
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self._serial = self.setup_serial()

    def setup_serial(self):
        if not HAS_SERIAL:
            self.get_logger().warn("PySerial missing. Drop disabled.")
            return None
        try:
            s = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.05)
            self.get_logger().info(f"🔌 Serial connected on {SERIAL_PORT}")
            return s
        except Exception as e:
            self.get_logger().warn(f"⚠️ Serial offline: {e}")
            return None

    def stop_motors(self):
        """Forces the robot to stop."""
        self.cmd_pub.publish(Twist())
        time.sleep(0.5)

    def drive_blind(self, linear_speed, duration_sec, action_name="Driving"):
        """Publishes a constant linear velocity for X seconds."""
        self.get_logger().info(f"🚗 {action_name} at {linear_speed}m/s for {duration_sec}s...")
        cmd = Twist()
        cmd.linear.x = float(linear_speed)
        
        t0 = time.time()
        while time.time() - t0 < duration_sec:
            self.cmd_pub.publish(cmd)
            time.sleep(0.05) # Loop rate
            
        self.stop_motors()

    def turn_blind(self, angular_speed, duration_sec, direction="left"):
        """Publishes a constant angular velocity for X seconds."""
        self.get_logger().info(f"🔄 Turning {direction} for {duration_sec}s...")
        cmd = Twist()
        # Positive is left (counter-clockwise), negative is right
        cmd.angular.z = float(angular_speed) if direction == "left" else -float(angular_speed)
        
        t0 = time.time()
        while time.time() - t0 < duration_sec:
            self.cmd_pub.publish(cmd)
            time.sleep(0.05)
            
        self.stop_motors()

    def drop_cargo(self):
        """Fires the serial command while completely stationary."""
        self.get_logger().info("📦 Firing Drop Command...")
        if self._serial:
            try:
                self._serial.write(CMD_DROP)
                self._serial.flush()
                self.get_logger().info("✅ Drop command sent (0x02)!")
            except Exception as e:
                self.get_logger().error(f"❌ Serial write failed: {e}")
        else:
            self.get_logger().info("⚠️ Simulating drop (No serial connection)...")
        
        # Wait for the mechanism to finish actuating
        time.sleep(2.5)

def main():
    rclpy.init()
    
    # 1. Grab Lane Choice
    setup_node = rclpy.create_node('lane_selector')
    setup_node.declare_parameter('lane', 'left_open')
    lane_choice = setup_node.get_parameter('lane').value
    setup_node.destroy_node()

    # Determine spin direction to avoid hitting the center wall
    spin_dir = "right" if lane_choice == "left_open" else "left"

    robot = DeadReckoner()
    
    print(f"\n🚀 STARTING DEAD RECKONING MISSION: {lane_choice.upper()}\n")
    
    # Give the user 2 seconds to back away after launching
    time.sleep(2.0)

    # ── PHASE 1: Outbound ──
    robot.drive_blind(CRUISE_SPEED, DRIVE_FWD_TIME, action_name="Heading to Port")
    
    # ── PHASE 2: Drop ──
    robot.drop_cargo()
    
    # ── PHASE 3: Back Up & Turn Around ──
    robot.drive_blind(-CRUISE_SPEED, DRIVE_REV_TIME, action_name="Reversing from Port")
    robot.turn_blind(TURN_SPEED, TURN_180_TIME, direction=spin_dir)
    
    # ── PHASE 4: Inbound ──
    robot.drive_blind(CRUISE_SPEED, DRIVE_HOME_TIME, action_name="Heading Home")

    print("\n🎉 MISSION COMPLETE! (Open-Loop)")

    if robot._serial:
        robot._serial.close()
    robot.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
