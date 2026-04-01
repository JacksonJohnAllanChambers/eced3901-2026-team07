#!/usr/bin/env python3
"""
fullest_load_right.py — Right Open Water: straight-in basket pickup
====================================================================
Sequence:
  Phase 1: Nav2 drives from home to port approach zone
  Phase 2: Drive straight at the port until front lidar triggers
  Phase 3: Turn CW 180° so back faces the cargo
  Phase 4: Reverse in until rear lidar detects cargo proximity
  Phase 5: BASKET GRAB (0x04)

Usage:
  Terminal 1: ros2 launch dalmotor robot.launch.py
  Terminal 2: ros2 launch eced3901 arena_real_amcl.launch.py lane:=right_open
  Terminal 3: ros2 run eced3901 fullest_load_right.py
"""

import math
import subprocess
import time

import numpy as np

try:
    import serial as _serial_mod
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
import tf2_ros

from nav2_simple_commander.robot_navigator import BasicNavigator

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

HOME_X   = 0.08
HOME_Y   = -1.08
HOME_YAW = 0.0

APPROACH_X   = 3.40
APPROACH_Y   = -1.08
APPROACH_YAW = 0.0

NAV2_PREEMPT_DIST = 0.25   # cancel Nav2 when this close to approach goal

# ── Phase 2: drive straight at port ──────────────────────────
DRIVE_SPEED          = 0.08   # m/s
DRIVE_FRONT_STOP     = 0.40   # stop driving forward when front lidar <= this
DRIVE_TIMEOUT        = 15.0   # safety timeout in seconds

# ── Phase 3: 180° CW turn ────────────────────────────────────
TURN_SPEED  = 0.40            # rad/s
TURN_TIME   = 7.9             # seconds — 180° at 0.40 rad/s (π / 0.40)

# ── Phase 4: reverse into cargo ──────────────────────────────
REVERSE_SPEED        = 0.06   # m/s (slow for precision)
REVERSE_REAR_STOP    = 0.15   # stop reversing when rear lidar <= this
REVERSE_TIMEOUT      = 10.0   # safety timeout in seconds

# ── Phase 5: basket grab ─────────────────────────────────────
BASKET_WAIT = 1.5             # seconds after sending 0x04

# ── Serial ───────────────────────────────────────────────────
SERIAL_PORTS    = ['/dev/ttyUSB0', '/dev/ttyUSB4']
SERIAL_BAUD     = 9600
CMD_BASKET_GRAB = b'\x04'

DT = 1.0 / 30.0


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
#  MISSION NODE
# ═══════════════════════════════════════════════════════════════

class MissionNode(Node):
    def __init__(self):
        super().__init__('fullest_load_right')
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.last_scan = None
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, scan_qos)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._serial = None
        self._open_serial()

    # ── Callbacks ────────────────────────────────────────────
    def _scan_cb(self, msg):
        self.last_scan = msg

    # ── ROS spin ─────────────────────────────────────────────
    def spin_ros(self, secs=0.1):
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.05)

    def stop(self):
        self.cmd_pub.publish(Twist())

    # ── AMCL pose ────────────────────────────────────────────
    def get_amcl_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3))
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return x, y, math.atan2(siny, cosy)
        except Exception:
            return None

    def print_pose(self, label=''):
        p = self.get_amcl_pose()
        li = self._lidar_summary()
        prefix = f'  [{label}] ' if label else '  '
        if p:
            print(f'{prefix}AMCL=({p[0]:.3f}, {p[1]:.3f}) yaw={math.degrees(p[2]):.1f}°'
                  f'  lidar F={li["front"]:.2f} R={li["rear"]:.2f}')
        else:
            print(f'{prefix}AMCL: unavailable')

    # ── Lidar ────────────────────────────────────────────────
    def _sector_min(self, start_deg, end_deg):
        scan = self.last_scan
        if scan is None:
            return float('inf')
        n = len(scan.ranges)
        rmin = float('inf')
        for deg in range(start_deg, end_deg + 1):
            v = scan.ranges[deg % n]
            if scan.range_min < v < scan.range_max:
                rmin = min(rmin, v)
        return rmin

    def get_front(self):
        return min(self._sector_min(0, 15), self._sector_min(345, 359))

    def get_rear(self):
        return self._sector_min(165, 195)

    def _lidar_summary(self):
        return {'front': self.get_front(), 'rear': self.get_rear()}

    # ── Serial ───────────────────────────────────────────────
    def _open_serial(self):
        if not HAS_SERIAL:
            self.get_logger().warn('pyserial not available')
            return
        for port in SERIAL_PORTS:
            try:
                subprocess.run(
                    ['stty', '-F', port,
                     '9600', 'cs8', '-cstopb', '-parenb', 'raw', '-hupcl'],
                    check=True, capture_output=True)
            except Exception:
                continue
            try:
                self._serial = _serial_mod.Serial()
                self._serial.port = port
                self._serial.baudrate = SERIAL_BAUD
                self._serial.bytesize = 8
                self._serial.parity = 'N'
                self._serial.stopbits = 1
                self._serial.timeout = 0.1
                self._serial.dsrdtr = False
                self._serial.rtscts = False
                self._serial.dtr = False   # suppress Arduino reset pulse on open
                self._serial.open()
                self.get_logger().info(f'Serial open: {port}')
                return
            except Exception:
                continue
        self.get_logger().warn(f'Serial: failed all ports {SERIAL_PORTS}')

    def send_serial(self, data, label):
        self.get_logger().info(f'SERIAL → {label}')
        if self._serial:
            try:
                self._serial.write(data)
            except Exception as e:
                self.get_logger().error(f'Serial write: {e}')
        else:
            self.get_logger().warn(f'No serial — {label} skipped')

    # ── Motion primitives ────────────────────────────────────
    def drive_until_front(self, speed, lidar_stop, timeout):
        """Drive forward until front lidar <= lidar_stop or timeout."""
        cmd = Twist()
        cmd.linear.x = float(speed)
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.cmd_pub.publish(cmd)
            self.spin_ros(0.05)
            time.sleep(DT)
            d = self.get_front()
            print(f'  front lidar: {d:.3f}m   ', end='\r')
            if d <= lidar_stop:
                print(f'\n  Front stop at {d:.3f}m')
                break
        else:
            print(f'\n  Drive timeout ({timeout}s)')
        self.stop()
        time.sleep(0.3)

    def reverse_until_rear(self, speed, lidar_stop, timeout):
        """Reverse until rear lidar <= lidar_stop or timeout."""
        cmd = Twist()
        cmd.linear.x = -abs(float(speed))
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.cmd_pub.publish(cmd)
            self.spin_ros(0.05)
            time.sleep(DT)
            d = self.get_rear()
            print(f'  rear lidar: {d:.3f}m   ', end='\r')
            if d <= lidar_stop:
                print(f'\n  Rear stop at {d:.3f}m')
                break
        else:
            print(f'\n  Reverse timeout ({timeout}s)')
        self.stop()
        time.sleep(0.3)

    def timed_turn(self, angular_z, duration):
        """Turn in place. angular_z > 0 = CCW, angular_z < 0 = CW."""
        cmd = Twist()
        cmd.angular.z = float(angular_z)
        t0 = time.time()
        while time.time() - t0 < duration:
            self.cmd_pub.publish(cmd)
            self.spin_ros(0.05)
            time.sleep(DT)
        self.stop()
        time.sleep(0.3)


# ═══════════════════════════════════════════════════════════════
#  MAIN SEQUENCE
# ═══════════════════════════════════════════════════════════════

def main():
    rclpy.init()

    nav = BasicNavigator()
    m = MissionNode()

    for _ in range(20):
        rclpy.spin_once(m, timeout_sec=0.1)

    print()
    print('══════════════════════════════════════════════════')
    print('   RIGHT OPEN WATER — STRAIGHT-IN BASKET PICKUP')
    print('══════════════════════════════════════════════════')
    print()

    print('Setting initial pose...')
    nav.setInitialPose(create_pose(nav, HOME_X, HOME_Y, HOME_YAW))
    nav.waitUntilNav2Active(localizer='amcl')
    print('Nav2 + AMCL ready!\n')
    m.print_pose('START')

    # ══════════════════════════════════════════════════════════
    #  PHASE 1 — Nav2 to port approach
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 1: Nav2 → port approach ═══')
    nav.goToPose(create_pose(nav, APPROACH_X, APPROACH_Y, APPROACH_YAW))

    while True:
        rclpy.spin_once(m, timeout_sec=0.05)
        if nav.isTaskComplete():
            print(f'  Nav2 result: {nav.getResult()}')
            break
        fb = nav.getFeedback()
        if fb:
            dist = fb.distance_remaining
            print(f'  dist remaining: {dist:.2f}m   ', end='\r')
            if dist < NAV2_PREEMPT_DIST:
                print(f'\n  Close enough ({dist:.2f}m) — cancelling Nav2')
                nav.cancelTask()
                break

    time.sleep(0.5)
    m.print_pose('NAV2 HANDOFF')

    # ══════════════════════════════════════════════════════════
    #  PHASE 2 — Drive straight at port until close
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 2: Drive straight at port ═══')
    m.drive_until_front(DRIVE_SPEED, DRIVE_FRONT_STOP, DRIVE_TIMEOUT)
    m.print_pose('AT PORT')

    # ══════════════════════════════════════════════════════════
    #  PHASE 3 — Turn CW 180° so back faces cargo
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 3: Turn CW 180° ═══')
    m.timed_turn(-TURN_SPEED, TURN_TIME)
    m.print_pose('FACING AWAY')

    # ══════════════════════════════════════════════════════════
    #  PHASE 4 — Reverse into cargo until close
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 4: Reverse into cargo ═══')
    m.reverse_until_rear(REVERSE_SPEED, REVERSE_REAR_STOP, REVERSE_TIMEOUT)
    m.print_pose('AT CARGO')

    # ══════════════════════════════════════════════════════════
    #  PHASE 5 — Basket grab
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 5: Basket grab ═══')
    m.send_serial(CMD_BASKET_GRAB, 'BASKET GRAB (0x04)')
    print('  Basket deployed.')
    m.spin_ros(BASKET_WAIT)
    m.print_pose('GRABBED')

    print()
    print('══════════════════════════════════════════════════')
    print('   PICKUP COMPLETE')
    print('══════════════════════════════════════════════════')
    print()

    m.stop()
    if m._serial:
        m._serial.close()
    m.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
