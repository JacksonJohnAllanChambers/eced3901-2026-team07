#!/usr/bin/env python3
"""
open_water_drop_fallback.py — Right Open Water: drop-only fallback
==================================================================
Simplified sequence:
  Phase 1: Nav2 drives from start to port approach zone
  Phase 2: Blind drive forward to drop position
  Phase 3: Drop cargo (0x02)
  Phase 4: Reverse out of port
  Phase 5: Nav2 drives home

Drop waypoint from recording: right_open_20260325_014422
  AMCL=(3.60, -0.97) yaw≈4°  lidar: F=1.24 Ri=0.59

Usage:
  Terminal 1:  ros2 launch dalmotor robot.launch.py
  Terminal 2:  ros2 launch eced3901 arena_real_amcl.launch.py lane:=right_open
  Terminal 3:  ros2 run eced3901 open_water_drop_fallback.py
"""

import math
import subprocess
import sys
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

# ── Phase 2: blind drive to drop zone ───────────────────────
DRIVE_SPEED = 0.08   # m/s
DRIVE_TIME       = 1.5  # seconds — tune this to control drop position (~0.08m/s)
DRIVE_LIDAR_STOP = 0.15  # emergency only — no wall ahead in open water

# ── Phase 3: drop ────────────────────────────────────────────
DROP_WAIT = 2.5      # seconds after drop command

# ── Phase 4: reverse ─────────────────────────────────────────
REVERSE_SPEED    = 0.08
REVERSE_TIME     = 4.0
REVERSE_LIDAR_STOP = 0.12   # rear safety

# ── Phase 4.5: turn to face home ─────────────────────────────
TURN_SPEED = 0.40   # rad/s
TURN_TIME  = 7.9    # seconds — 180° CW at 0.40 rad/s (π / 0.40)

# ── Serial ───────────────────────────────────────────────────
SERIAL_PORTS = ['/dev/ttyUSB0', '/dev/ttyUSB4']
SERIAL_BAUD  = 9600
CMD_DROP     = b'\x02'

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
        super().__init__('open_water_drop_fallback')
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
        prefix = f'  [{label}] ' if label else '  '
        li = self._lidar_summary()
        if p:
            print(f'{prefix}AMCL=({p[0]:.3f}, {p[1]:.3f}) yaw={math.degrees(p[2]):.1f}°'
                  f'  lidar F={li["front"]:.2f} R={li["rear"]:.2f}')
        else:
            print(f'{prefix}AMCL: unavailable')

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

    def timed_turn(self, direction='cw'):
        """Turn 180° in place. direction: 'ccw' (+z) or 'cw' (-z)."""
        self.get_logger().info(f'TURN 180° {direction.upper()}')
        cmd = Twist()
        cmd.angular.z = TURN_SPEED if direction == 'ccw' else -TURN_SPEED
        t0 = time.time()
        while time.time() - t0 < TURN_TIME:
            self.cmd_pub.publish(cmd)
            self.spin_ros(0.05)
            time.sleep(DT)
        self.stop()
        time.sleep(0.3)

    def blind_drive(self, speed, duration, lidar_field=None, lidar_stop=None):
        self.get_logger().info(
            f'DRIVE {speed:+.2f} m/s for {duration:.1f}s'
            + (f', stop if {lidar_field}<{lidar_stop}m' if lidar_stop else ''))
        cmd = Twist()
        cmd.linear.x = float(speed)
        t0 = time.time()
        while time.time() - t0 < duration:
            self.cmd_pub.publish(cmd)
            self.spin_ros(0.05)
            time.sleep(DT)
            if lidar_stop is not None and lidar_field is not None:
                d = self.get_front() if lidar_field == 'front' else self.get_rear()
                if d < lidar_stop:
                    self.get_logger().info(f'  lidar {lidar_field} stop at {d:.3f}m')
                    break
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
    print('   RIGHT OPEN WATER — DROP FALLBACK')
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

    while not nav.isTaskComplete():
        fb = nav.getFeedback()
        if fb:
            dist = fb.distance_remaining
            print(f'  dist remaining: {dist:.2f}m   ', end='\r')
            if dist < NAV2_PREEMPT_DIST:
                print(f'\n  Close enough ({dist:.2f}m) — cancelling Nav2')
                nav.cancelTask()
                break

    print(f'  Nav2 result: {nav.getResult()}')
    time.sleep(0.5)
    m.print_pose('NAV2 HANDOFF')

    # ══════════════════════════════════════════════════════════
    #  PHASE 2 — Blind drive to drop zone
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 2: Drive to drop zone ═══')
    m.blind_drive(DRIVE_SPEED, DRIVE_TIME)
    m.print_pose('DROP POSITION')

    # ══════════════════════════════════════════════════════════
    #  PHASE 3 — Drop cargo
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 3: Drop cargo ═══')
    m.send_serial(CMD_DROP, 'DROP (0x02)')
    print('  Waiting for cargo to release...')
    t0 = time.time()
    while time.time() - t0 < DROP_WAIT:
        m.spin_ros(0.1)

    # ══════════════════════════════════════════════════════════
    #  PHASE 4 — Reverse out
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 4: Reverse out ═══')
    m.blind_drive(-REVERSE_SPEED, REVERSE_TIME,
                  lidar_field='rear', lidar_stop=REVERSE_LIDAR_STOP)
    m.print_pose('REVERSED')

    # ══════════════════════════════════════════════════════════
    #  PHASE 4.5 — Turn CW to face home
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 4.5: Turn CCW 180° ═══')
    m.timed_turn(direction='ccw')
    m.print_pose('TURNED')

    # ══════════════════════════════════════════════════════════
    #  PHASE 5 — Nav2 home
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 5: Nav2 → home ═══')
    nav.goToPose(create_pose(nav, HOME_X, HOME_Y, 180.0))

    while not nav.isTaskComplete():
        fb = nav.getFeedback()
        if fb:
            dist = fb.distance_remaining
            print(f'  dist remaining: {dist:.2f}m   ', end='\r')

    print(f'\n  Nav2 result: {nav.getResult()}')
    m.print_pose('HOME')

    print('\n══════════════════════════════════════════════════')
    print('   MISSION COMPLETE')
    print('══════════════════════════════════════════════════\n')

    m.stop()
    if m._serial:
        m._serial.close()
    m.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
