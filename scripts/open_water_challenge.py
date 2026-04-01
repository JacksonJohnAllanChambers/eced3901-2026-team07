#!/usr/bin/env python3
"""
open_water_challenge.py — Right-open route from recorded marks
================================================================
Uses the latest recorder route (20260325_063331) and executes:
  - lane/midfield approach
  - turn into right port and back-in sideways
  - basket grab (pickup command)
  - rotate/align then cargo drop command
  - reverse out, turn, and return toward home
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

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


# ── Start (from recording coords_ref) ─────────────────────────────────
HOME_X = 0.08
HOME_Y = -1.08
HOME_YAW = 0.0

# ── Route marks from open_water_recording_right_open_20260325_063331 ──
ROUTE_POSES = [
    ("mark_1",      3.0628, -0.9520, -10.1),
    ("mark_2",      3.2073, -0.9758, -9.7),
    ("mark_3",      3.4123, -0.9583, 38.5),
    ("mark_4",      3.5020, -0.9116, 93.9),
    ("mark_5",      3.5023, -0.9380, 109.2),
    ("basket_grab", 3.5022, -0.9339, 108.0),
    ("cargo_drop",  3.5025, -0.9244, 37.2),
    ("mark_6",      2.9059, -0.8467, -147.6),
    ("mark_7",      0.1837, -1.1182, -36.8),
]

# ── Serial ─────────────────────────────────────────────────────────────
SERIAL_PORTS = ['/dev/ttyUSB0', '/dev/ttyUSB4']
SERIAL_BAUD = 9600
CMD_PICKUP = b'\x01'
CMD_DROP = b'\x02'

# ── Motion actions between marks ───────────────────────────────────────
BACKOUT_SPEED = 0.08
BACKOUT_TIME = 1.2
BACKOUT_REAR_STOP = 0.12
DT = 1.0 / 30.0


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


class MissionNode(Node):
    def __init__(self):
        super().__init__('open_water_mission')
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.last_scan = None
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, scan_qos)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._serial = None
        self._open_serial()

    def _scan_cb(self, msg):
        self.last_scan = msg

    def spin_ros(self, secs=0.1):
        t0 = time.time()
        while time.time() - t0 < secs:
            rclpy.spin_once(self, timeout_sec=0.05)

    def stop(self):
        self.cmd_pub.publish(Twist())

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
        if p:
            print(f'  [{label}] AMCL=({p[0]:.4f}, {p[1]:.4f}) yaw={math.degrees(p[2]):.1f}°')

    def _sector_min(self, start_deg, end_deg):
        scan = self.last_scan
        if scan is None:
            return float('inf')
        n = len(scan.ranges)
        if n == 0:
            return float('inf')
        rmin = float('inf')
        for deg in range(start_deg, end_deg + 1):
            v = scan.ranges[deg % n]
            if scan.range_min < v < scan.range_max:
                rmin = min(rmin, v)
        return rmin

    def get_rear(self):
        return self._sector_min(165, 195)

    def blind_reverse(self, speed, duration, rear_stop):
        self.get_logger().info(f'REVERSE {duration:.1f}s at {speed:.2f} m/s')
        cmd = Twist()
        cmd.linear.x = -abs(speed)
        t0 = time.time()
        while time.time() - t0 < duration:
            self.cmd_pub.publish(cmd)
            self.spin_ros(0.05)
            time.sleep(DT)
            if self.get_rear() < rear_stop:
                self.get_logger().info(f'  rear lidar stop: {self.get_rear():.3f}m')
                break
        self.stop()
        time.sleep(0.3)

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
                self._serial = _serial_mod.Serial(
                    port, SERIAL_BAUD,
                    bytesize=8, parity='N', stopbits=1, timeout=0.1,
                    dsrdtr=False, rtscts=False)
                self._serial.dtr = False
                self.get_logger().info(f'Serial open: {port}')
                return
            except Exception:
                continue
        self.get_logger().warn(f'Serial failed all ports: {SERIAL_PORTS}')

    def send_serial(self, data, label):
        self.get_logger().info(f'SERIAL → {label}')
        if self._serial:
            try:
                self._serial.write(data)
            except Exception as e:
                self.get_logger().error(f'Serial write failed: {e}')


def nav_to_pose(nav, pose, label):
    print(f'\n→ NAV TO {label}')
    nav.goToPose(pose)
    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback:
            print(f'  {label}: {feedback.distance_remaining:.2f}m   ', end='\r')
    result = nav.getResult()
    print(f'  {label} result: {result}')
    return result


def main():
    rclpy.init()

    nav = BasicNavigator()
    m = MissionNode()
    for _ in range(20):
        rclpy.spin_once(m, timeout_sec=0.1)

    print('\n════════════════════════════════════════════')
    print('  RIGHT OPEN — RECORDED ROUTE AUTONOMY')
    print('════════════════════════════════════════════\n')

    nav.setInitialPose(create_pose(nav, HOME_X, HOME_Y, HOME_YAW))
    nav.waitUntilNav2Active(localizer='amcl')
    m.print_pose('START')

    for label, x, y, yaw in ROUTE_POSES:
        result = nav_to_pose(nav, create_pose(nav, x, y, yaw), label)
        if result != TaskResult.SUCCEEDED:
            print(f'\nMission abort: could not reach {label}')
            m.stop()
            if m._serial:
                m._serial.close()
            m.destroy_node()
            rclpy.shutdown()
            return

        m.print_pose(label)

        if label == 'basket_grab':
            m.send_serial(CMD_PICKUP, 'BASKET_GRAB/PICKUP (0x01)')
            time.sleep(1.5)

        if label == 'cargo_drop':
            m.send_serial(CMD_DROP, 'CARGO_DROP (0x02)')
            time.sleep(2.0)

            print('  Backing out a bit...')
            m.blind_reverse(BACKOUT_SPEED, BACKOUT_TIME, BACKOUT_REAR_STOP)

    print('\n════════════════════════════════════════════')
    print('  RECORDED ROUTE COMPLETE')
    print('════════════════════════════════════════════\n')

    m.stop()
    if m._serial:
        m._serial.close()
    m.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
