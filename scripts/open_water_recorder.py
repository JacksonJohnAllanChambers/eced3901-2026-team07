#!/usr/bin/env python3
"""
open_water_recorder.py — Calibration run for open water challenge
=================================================================
Phase 1 (AUTO):  Nav2 drives to the port approach point (same as simple_open_water.py)
Phase 2 (MANUAL): Teleop with WASD. Poses are recorded automatically when you
                   trigger serial actions (drop / pickup / basket grab / release).

Controls:
  W/S/A/D  — drive fwd / rev / rotate left / rotate right
  X        — stop

  E  — DROP cargo      (sends 0x02 over UART, records "cargo_drop" pose)
  R  — PICKUP cargo    (sends 0x01 over UART, records "cargo_pickup" pose)
  F  — BASKET GRAB     (sends 0x04 over UART, records "basket_grab" pose)
  G  — BASKET RELEASE  (sends 0x03 over UART, records "basket_release" pose)

  M  — mark current pose (generic waypoint, auto-numbered)
  P  — print current AMCL pose + lidar
  Q  — save recording & quit

Auto-recorded: nav2_handoff (when Nav2 stops), continuous 2Hz trace with
               AMCL pose + lidar + any lifeboat CV detections.

Output: ~/ros2_ws/open_water_recording_<lane>_<timestamp>.json

Usage:
  ros2 run eced3901 open_water_recorder --ros-args -p lane:=right_open
  ros2 run eced3901 open_water_recorder --ros-args -p lane:=left_open
"""

import json
import math
import os
import subprocess
import sys
import select
import termios
import time
import tty
from datetime import datetime

import numpy as np

try:
    import serial as _serial_mod
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String
import tf2_ros

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# ─── Serial ─────────────────────────────────────────────────────────
SERIAL_PORTS  = ['/dev/ttyUSB0', '/dev/ttyUSB4']  # Arduino moves between these
SERIAL_BAUD   = 9600
CMD_PICKUP    = b'\x01'
CMD_DROP      = b'\x02'
CMD_LIFEBOAT_RELEASE  = b'\x03'
CMD_LIFEBOAT_CAPTURE  = b'\x04'

# ─── Teleop speeds ──────────────────────────────────────────────────
TELEOP_LINEAR  = 0.10   # m/s
TELEOP_ANGULAR = 0.35   # rad/s

# ─── Nav2 open water coordinates (same as simple_open_water.py) ─────
OPEN_WATER_COORDS = {
    'left_open': {
        'x': 0.031,
        'y': -0.009,
        'goal_x': 3.40,
        'home_yaw': 0.0,
    },
    'right_open': {
        'x': 0.08,
        'y': -1.08,
        'goal_x': 3.40,
        'home_yaw': 0.0,
    },
}


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


class Recorder(Node):
    def __init__(self, lane: str):
        super().__init__('open_water_recorder')
        self.lane = lane
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # TF for AMCL pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscriptions
        self.last_odom = None
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.last_imu = None
        self.create_subscription(Imu, '/imu/imu', self._imu_cb, 10)
        self.last_scan = None
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, scan_qos)
        self.last_detections = []
        self.last_det_time = 0.0
        self.create_subscription(String, '/cv/detections', self._det_cb, 10)

        # Recording state
        self.waypoints = []
        self.trace = []
        self.mark_count = 0
        self.last_trace_time = 0.0

        # Serial — open all available ports (cargo + lifeboat may be on different Arduinos)
        self._serials = {}
        self._open_all_serial()

    # ─── Callbacks ──────────────────────────────────────────────
    def _odom_cb(self, msg):  self.last_odom = msg
    def _imu_cb(self, msg):   self.last_imu = msg
    def _scan_cb(self, msg):  self.last_scan = msg
    def _det_cb(self, msg):
        try:
            self.last_detections = json.loads(msg.data)
            self.last_det_time = time.time()
        except json.JSONDecodeError:
            self.last_detections = []

    # ─── Serial ─────────────────────────────────────────────────
    def _port_in_use(self, port):
        """Check if another process has this port open (fuser)."""
        try:
            r = subprocess.run(['fuser', port], capture_output=True, timeout=2)
            return r.returncode == 0  # 0 = someone has it open
        except Exception:
            return False

    def _open_all_serial(self):
        if not _SERIAL_OK:
            self.get_logger().warn('pyserial not installed — serial disabled')
            return
        for port in SERIAL_PORTS:
            if self._port_in_use(port):
                self.get_logger().info(f'Serial skip {port} — in use by another process')
                continue
            try:
                subprocess.run(
                    ['stty', '-F', port,
                     '9600', 'cs8', '-cstopb', '-parenb', 'raw', '-hupcl'],
                    check=True, capture_output=True)
            except Exception:
                continue
            try:
                ser = _serial_mod.Serial(
                    port, SERIAL_BAUD,
                    bytesize=8, parity='N', stopbits=1, timeout=0.1,
                    dsrdtr=False, rtscts=False)
                ser.dtr = False
                self._serials[port] = ser
                self.get_logger().info(f'Serial open: {port}')
            except Exception:
                continue
        if not self._serials:
            self.get_logger().warn(f'Serial: no free ports in {SERIAL_PORTS}')

    def send_serial(self, data: bytes, label: str):
        if not self._serials:
            self.get_logger().warn(f'No serial ports — {label} skipped')
            return
        for port, ser in self._serials.items():
            try:
                ser.write(data)
                self.get_logger().info(f'Serial {port} → {label}')
            except Exception as e:
                self.get_logger().error(f'Serial {port} write failed: {e}')

    # ─── Pose helpers ───────────────────────────────────────────
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
            return (x, y, math.atan2(siny, cosy))
        except Exception:
            return None

    def lidar_summary(self):
        def _min(s, e):
            scan = self.last_scan
            if scan is None: return float('inf')
            n = len(scan.ranges)
            if n == 0: return float('inf')
            r = float('inf')
            for d in range(s, e + 1):
                v = scan.ranges[d % n]
                if scan.range_min < v < scan.range_max:
                    r = min(r, v)
            return r
        return {
            'front': min(_min(0, 15), _min(345, 359)),
            'rear':  _min(165, 195),
            'left':  _min(40, 120),
            'right': _min(240, 320),
        }

    # ─── Recording ──────────────────────────────────────────────
    def record_waypoint(self, label):
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)
        amcl = self.get_amcl_pose()
        lidar = self.lidar_summary()
        if amcl is None:
            print(f'  !! No AMCL pose — "{label}" NOT recorded')
            return
        wp = {
            'label': label,
            'time': time.time(),
            'amcl': {'x': round(amcl[0], 4), 'y': round(amcl[1], 4),
                     'yaw_rad': round(amcl[2], 4), 'yaw_deg': round(math.degrees(amcl[2]), 1)},
            'lidar': {k: round(v, 3) for k, v in lidar.items()},
        }
        self.waypoints.append(wp)
        print(f'  >> RECORDED "{label}"  ({amcl[0]:.4f}, {amcl[1]:.4f}) yaw={math.degrees(amcl[2]):.1f}°')

    def record_trace(self):
        now = time.time()
        if now - self.last_trace_time < 0.5:
            return
        self.last_trace_time = now
        amcl = self.get_amcl_pose()
        if amcl is None:
            return
        lidar = self.lidar_summary()
        lb = None
        if now - self.last_det_time < 1.0:
            for d in self.last_detections:
                if d.get('label') == 'lifeboat' and d.get('confidence', 0) > 0.35:
                    lb = {'conf': d['confidence'], 'center': d['center'], 'area': d.get('area', 0)}
                    break
        self.trace.append({
            't': round(now, 3), 'x': round(amcl[0], 4), 'y': round(amcl[1], 4),
            'yaw': round(math.degrees(amcl[2]), 1),
            'lidar_f': round(lidar['front'], 3), 'lidar_r': round(lidar['rear'], 3),
            'lifeboat': lb,
        })

    def print_pose(self):
        amcl = self.get_amcl_pose()
        lidar = self.lidar_summary()
        if amcl:
            print(f'  AMCL: ({amcl[0]:.4f}, {amcl[1]:.4f}) yaw={math.degrees(amcl[2]):.1f}°')
        else:
            print(f'  AMCL: NOT AVAILABLE')
        print(f'  Lidar: F={lidar["front"]:.2f} R={lidar["rear"]:.2f} L={lidar["left"]:.2f} Ri={lidar["right"]:.2f}')

    def save(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fn = os.path.expanduser(f'~/ros2_ws/open_water_recording_{self.lane}_{ts}.json')
        with open(fn, 'w') as f:
            json.dump({
                'lane': self.lane, 'timestamp': ts,
                'coords_ref': OPEN_WATER_COORDS.get(self.lane, {}),
                'waypoints': self.waypoints,
                'trace': self.trace,
            }, f, indent=2)
        print(f'\n  Saved: {fn}')
        print(f'  {len(self.waypoints)} waypoints, {len(self.trace)} trace points')
        return fn

    def stop(self):
        self.cmd_pub.publish(Twist())


class KeyReader:
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
    def __enter__(self):
        tty.setraw(self.fd)
        return self
    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)
    def get(self):
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
        return None


def main():
    rclpy.init()

    # ── Parse lane ──────────────────────────────────────────────────
    lane = 'right_open'
    if len(sys.argv) > 1 and sys.argv[1] in OPEN_WATER_COORDS:
        lane = sys.argv[1]
    else:
        pn = rclpy.create_node('_rec_param')
        pn.declare_parameter('lane', 'right_open')
        lane = pn.get_parameter('lane').value
        pn.destroy_node()

    if lane not in OPEN_WATER_COORDS:
        print(f'Invalid lane "{lane}". Use: {list(OPEN_WATER_COORDS.keys())}')
        rclpy.shutdown()
        return

    data = OPEN_WATER_COORDS[lane]

    # ── Phase 1: Nav2 drives to port approach ──────────────────────
    print(f'\n=== Open Water Recorder — {lane.upper()} ===')
    nav = BasicNavigator()

    print('Setting AMCL initial pose...')
    nav.setInitialPose(create_pose(nav, data['x'], data['y'], data['home_yaw']))

    print('Waiting for Nav2/AMCL...')
    nav.waitUntilNav2Active(localizer='amcl')
    print('Nav2 active!')

    print(f'Navigating to ({data["x"]:.3f}, {data["goal_x"]:.3f})...')
    nav.goToPose(create_pose(nav, data['goal_x'], data['y'], 0.0))

    while not nav.isTaskComplete():
        fb = nav.getFeedback()
        if fb:
            d = fb.distance_remaining
            print(f'  {d:.2f} m remaining   ', end='\r')
            if d < 0.40:
                print('\n  Cancelling Nav2 — switching to manual...')
                nav.cancelTask()
                break

    res = nav.getResult()
    print(f'  Nav2 result: {res}')
    time.sleep(0.5)

    # ── Phase 2: Manual teleop ─────────────────────────────────────
    rec = Recorder(lane)
    print('\nWaiting for AMCL TF...')
    for _ in range(30):
        rclpy.spin_once(rec, timeout_sec=0.1)

    # Auto-record where Nav2 handed off
    rec.record_waypoint('nav2_handoff')

    print()
    print('=' * 50)
    print('  W/S/A/D = drive    X = stop')
    print('  E = DROP cargo     R = PICKUP cargo')
    print('  F = BASKET GRAB    G = BASKET RELEASE')
    print('  M = mark waypoint  P = print pose')
    print('  Q = save & quit')
    print('=' * 50)

    cmd = Twist()

    with KeyReader() as kb:
        try:
            while rclpy.ok():
                rclpy.spin_once(rec, timeout_sec=0.01)
                rec.record_trace()

                key = kb.get()
                if key is None:
                    rec.cmd_pub.publish(cmd)
                    time.sleep(0.03)
                    continue

                key = key.lower()

                # Movement
                if key == 'w':
                    cmd = Twist(); cmd.linear.x = TELEOP_LINEAR
                elif key == 's':
                    cmd = Twist(); cmd.linear.x = -TELEOP_LINEAR
                elif key == 'a':
                    cmd = Twist(); cmd.angular.z = TELEOP_ANGULAR
                elif key == 'd':
                    cmd = Twist(); cmd.angular.z = -TELEOP_ANGULAR
                elif key in ('x', ' '):
                    cmd = Twist(); rec.stop()

                # Serial actions → auto-record pose
                elif key == 'e':
                    rec.stop(); cmd = Twist()
                    rec.send_serial(CMD_DROP, 'DROP')
                    print('  CARGO DROP sent (0x02)')
                    rec.record_waypoint('cargo_drop')
                elif key == 'r':
                    rec.stop(); cmd = Twist()
                    rec.send_serial(CMD_PICKUP, 'PICKUP')
                    print('  CARGO PICKUP sent (0x01)')
                    rec.record_waypoint('cargo_pickup')
                elif key == 'f':
                    rec.stop(); cmd = Twist()
                    rec.send_serial(CMD_LIFEBOAT_CAPTURE, 'BASKET GRAB')
                    print('  BASKET GRAB sent (0x04)')
                    rec.record_waypoint('basket_grab')
                elif key == 'g':
                    rec.stop(); cmd = Twist()
                    rec.send_serial(CMD_LIFEBOAT_RELEASE, 'BASKET RELEASE')
                    print('  BASKET RELEASE sent (0x03)')
                    rec.record_waypoint('basket_release')

                # Utilities
                elif key == 'm':
                    rec.stop(); cmd = Twist()
                    rec.mark_count += 1
                    rec.record_waypoint(f'mark_{rec.mark_count}')
                elif key == 'p':
                    rclpy.spin_once(rec, timeout_sec=0.1)
                    rec.print_pose()

                # Quit
                elif key in ('q', '\x03'):
                    rec.stop(); break

                rec.cmd_pub.publish(cmd)
                time.sleep(0.03)
        except KeyboardInterrupt:
            pass

    # ── Save ──────────────────────────────────────────────────────
    rec.stop()
    print('\n\nSaving...')
    fn = rec.save()
    print(f'\nTransfer: scp student@10.0.0.207:{fn} .')

    if rec.waypoints:
        print(f'\n  Waypoints:')
        for w in rec.waypoints:
            a = w['amcl']
            print(f'    {w["label"]:20s} ({a["x"]:.4f}, {a["y"]:.4f}) yaw={a["yaw_deg"]:.1f}°')
    print()

    rec.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
