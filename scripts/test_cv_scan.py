#!/usr/bin/env python3
"""
Test CV Scan — Isolated test for cargo/lifeboat detection.

Only requires:
  1. ros2 launch dalmotor robot.launch.py   (motor + lidar + IMU)
  2. ros2 launch dalibot_cv dalibot_cv_launch.py  (camera + detection)

NO AMCL / Nav2 needed.

Place the robot where it can see cargo/lifeboat, then run:
  ros2 run eced3901 test_cv_scan.py

The robot will:
  1. Rotate in place scanning for cargo
  2. Center on cargo and report orientation (sideways vs endwise)
  3. Rotate scanning for lifeboat
  4. Center on lifeboat and report detection
  5. Stop and print summary
"""

import json
import math
import time
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String, Bool

# ── CV constants ──
CV_CENTER_X     = 640
CV_CENTER_TOL   = 80
CV_ROTATE       = 0.20
CV_MIN_CONFIDENCE = 0.35
CV_SCAN_TIMEOUT = 15.0
CAM_WIDTH       = 1280
DT              = 1.0 / 30.0


class CVScanTest(Node):
    def __init__(self):
        super().__init__('cv_scan_test')

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/cv/nav_status', 10)

        self.last_odom = None
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.last_imu = None
        self.create_subscription(Imu, '/imu/imu', self._imu_cb, 10)
        self.last_scan = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.last_detections = []
        self.last_det_time = 0.0
        self.create_subscription(String, '/cv/detections', self._det_cb, 10)

        self.wall_ahead = False
        self.create_subscription(Bool, '/cv/wall_ahead', self._wall_cb, 10)

    def _odom_cb(self, msg):  self.last_odom = msg
    def _imu_cb(self, msg):   self.last_imu = msg
    def _scan_cb(self, msg):  self.last_scan = msg
    def _wall_cb(self, msg):  self.wall_ahead = msg.data

    def set_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(f'STATUS: {text}')

    def _det_cb(self, msg):
        try:
            self.last_detections = json.loads(msg.data)
            self.last_det_time = time.time()
        except json.JSONDecodeError:
            self.last_detections = []

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _get_yaw(self):
        if self.last_imu is not None:
            q = self.last_imu.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return math.atan2(siny, cosy)
        if self.last_odom is not None:
            q = self.last_odom.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return math.atan2(siny, cosy)
        return None

    @staticmethod
    def _normalize_angle(a):
        while a > math.pi:  a -= 2 * math.pi
        while a < -math.pi: a += 2 * math.pi
        return a

    def _spin_cv(self, duration=0.15):
        t0 = time.time()
        while time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _get_best_detection(self, label):
        if time.time() - self.last_det_time > 1.0:
            return None
        best = None
        for d in self.last_detections:
            if d.get('label') != label:
                continue
            if d.get('confidence', 0) < CV_MIN_CONFIDENCE:
                continue
            if best is None or d['confidence'] > best['confidence']:
                best = d
        return best

    # ─── Test behaviours ────────────────────────────────────────
    def scan_for(self, label, timeout=None):
        if timeout is None:
            timeout = CV_SCAN_TIMEOUT
        self.get_logger().info(f'SCAN: Looking for {label}...')
        t0 = time.time()
        cmd = Twist()
        total_rotated = 0.0
        last_yaw = None

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin_cv(0.1)

            det = self._get_best_detection(label)
            if det is not None:
                cx, cy = det['center']
                self.get_logger().info(
                    f'SCAN: Found {label}! conf={det["confidence"]:.2f} '
                    f'center=({cx},{cy}) area={det["area"]:.0f} '
                    f'aspect={det.get("aspect_ratio", 0):.2f}')
                self._stop()
                return det

            cmd.angular.z = CV_ROTATE
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)

            yaw = self._get_yaw()
            if yaw is not None:
                if last_yaw is not None:
                    total_rotated += abs(self._normalize_angle(yaw - last_yaw))
                last_yaw = yaw
                if total_rotated > 2 * math.pi + 0.5:
                    self.get_logger().warn(f'SCAN: Full rotation, {label} not found')
                    break
            time.sleep(DT)

        self._stop()
        self.get_logger().warn(f'SCAN: Timeout ({label})')
        return None

    def center_on(self, label, timeout=8.0):
        self.get_logger().info(f'CENTER: Centering on {label}...')
        t0 = time.time()
        cmd = Twist()

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin_cv(0.1)
            det = self._get_best_detection(label)
            if det is None:
                cmd.angular.z = CV_ROTATE * 0.5
                cmd.linear.x = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue

            offset = det['center'][0] - CV_CENTER_X
            if abs(offset) < CV_CENTER_TOL:
                self._stop()
                self.get_logger().info(
                    f'CENTER: {label} centered (offset={offset:.0f}px)')
                return det

            rot_speed = min(CV_ROTATE, max(0.05, abs(offset) / CAM_WIDTH * 0.6))
            cmd.angular.z = -rot_speed if offset > 0 else rot_speed
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)

        self._stop()
        self.get_logger().warn(f'CENTER: Timeout ({label})')
        return None

    def wait_for_cv(self, timeout=10.0):
        self.get_logger().info('Waiting for CV pipeline...')
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.last_det_time > 0:
                self.get_logger().info('CV pipeline active!')
                return True
        self.get_logger().error('CV pipeline not detected! Is dalibot_cv running?')
        return False


def main():
    rclpy.init()
    node = CVScanTest()

    if not node.wait_for_cv():
        node.destroy_node()
        rclpy.shutdown()
        return

    # Wait a moment for IMU/odom
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)

    results = {}

    # ── Test 1: Scan for cargo ──
    node.get_logger().info('\n========== TEST 1: CARGO SCAN ==========')
    node.set_status('Test 1: Scanning for cargo')
    cargo = node.scan_for('cargo')
    if cargo:
        node.set_status('Test 1: Centering on cargo')
        cargo = node.center_on('cargo')
    if cargo:
        aspect = cargo.get('aspect_ratio', 1.0)
        orientation = 'sideways' if aspect > 1.2 else 'endwise'
        results['cargo'] = {
            'found': True,
            'orientation': orientation,
            'aspect_ratio': aspect,
            'area': cargo['area'],
            'confidence': cargo['confidence'],
            'center': cargo['center'],
        }
        node.get_logger().info(
            f'CARGO RESULT: {orientation} (aspect={aspect:.2f}, '
            f'area={cargo["area"]:.0f}, conf={cargo["confidence"]:.2f})')
    else:
        results['cargo'] = {'found': False}
        node.get_logger().warn('CARGO RESULT: Not found')

    time.sleep(1.0)

    # ── Test 2: Scan for lifeboat ──
    node.get_logger().info('\n========== TEST 2: LIFEBOAT SCAN ==========')
    node.set_status('Test 2: Scanning for lifeboat')
    lifeboat = node.scan_for('lifeboat')
    if lifeboat:
        node.set_status('Test 2: Centering on lifeboat')
        lifeboat = node.center_on('lifeboat')
    if lifeboat:
        results['lifeboat'] = {
            'found': True,
            'area': lifeboat['area'],
            'confidence': lifeboat['confidence'],
            'center': lifeboat['center'],
            'aspect_ratio': lifeboat.get('aspect_ratio', 0),
        }
        node.get_logger().info(
            f'LIFEBOAT RESULT: area={lifeboat["area"]:.0f}, '
            f'conf={lifeboat["confidence"]:.2f}')
    else:
        results['lifeboat'] = {'found': False}
        node.get_logger().warn('LIFEBOAT RESULT: Not found')

    # ── Summary ──
    node.set_status('Scan test complete')
    node.get_logger().info('\n========== SCAN TEST SUMMARY ==========')
    for label, r in results.items():
        if r['found']:
            extras = ', '.join(f'{k}={v}' for k, v in r.items() if k != 'found')
            node.get_logger().info(f'  {label}: FOUND — {extras}')
        else:
            node.get_logger().info(f'  {label}: NOT FOUND')
    node.get_logger().info('========================================')

    node._stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
