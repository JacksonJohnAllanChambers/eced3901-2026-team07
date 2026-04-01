#!/usr/bin/env python3
"""
Test CV Lifeboat Approach — Isolated test for finding and driving to the lifeboat.

Only requires:
  1. ros2 launch dalmotor robot.launch.py   (motor + lidar + IMU)
  2. ros2 launch dalibot_cv dalibot_cv_launch.py  (camera + detection)

NO AMCL / Nav2 needed.

Place the robot where it can potentially see the lifeboat (within ~1-2m), then run:
  ros2 run eced3901 test_cv_lifeboat.py

The robot will:
  1. Rotate 360° scanning for the lifeboat (full rotation before giving up)
  2. Center on it in the camera frame
  3. Drive forward until the lifeboat leaves the frame
  4. Rotate 180° counterclockwise
  5. Stop and print result
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String, Bool

# ── Constants ──
CV_CENTER_X        = 640
CV_CENTER_TOL      = 80
CV_ROTATE          = 0.20
CV_LINEAR          = 0.10
CV_MIN_CONFIDENCE  = 0.35
CV_SCAN_TIMEOUT    = 15.0
CV_APPROACH_TIMEOUT = 20.0
CV_LIFEBOAT_AREA_CLOSE = 8000
CAM_WIDTH          = 1280
LIDAR_FRONT_STOP   = 0.12
LIDAR_SLOW_DIST    = 0.55
ROTATE_SPEED       = 0.30
DT                 = 1.0 / 30.0


class CVLifeboatTest(Node):
    def __init__(self):
        super().__init__('cv_lifeboat_test')

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

    def _lidar_sector_min(self, start_deg, end_deg):
        scan = self.last_scan
        if scan is None:
            return float('inf')
        n = len(scan.ranges)
        if n == 0:
            return float('inf')
        r_min = float('inf')
        for deg in range(start_deg, end_deg + 1):
            idx = deg % n
            r = scan.ranges[idx]
            if scan.range_min < r < scan.range_max:
                r_min = min(r_min, r)
        return r_min

    def _get_front_min(self):
        return min(self._lidar_sector_min(0, 15),
                   self._lidar_sector_min(345, 359))

    # ─── Test behaviours ────────────────────────────────────────
    def scan_for(self, timeout=None):
        if timeout is None:
            timeout = CV_SCAN_TIMEOUT
        self.get_logger().info('SCAN: Looking for lifeboat...')
        t0 = time.time()
        cmd = Twist()
        total_rotated = 0.0
        last_yaw = None

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin_cv(0.1)

            det = self._get_best_detection('lifeboat')
            if det is not None:
                cx, cy = det['center']
                self.get_logger().info(
                    f'SCAN: Found lifeboat! conf={det["confidence"]:.2f} '
                    f'center=({cx},{cy}) area={det["area"]:.0f}')
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
                if total_rotated > 2 * math.pi:
                    self.get_logger().warn('SCAN: Full 360° rotation, lifeboat not found')
                    break
            time.sleep(DT)

        self._stop()
        self.get_logger().warn('SCAN: Not found after full rotation')
        return None

    def center_on(self, timeout=8.0):
        self.get_logger().info('CENTER: Centering on lifeboat...')
        t0 = time.time()
        cmd = Twist()

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin_cv(0.1)
            det = self._get_best_detection('lifeboat')
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
                    f'CENTER: Lifeboat centered (offset={offset:.0f}px)')
                return det

            rot_speed = min(CV_ROTATE, max(0.05, abs(offset) / CAM_WIDTH * 0.6))
            cmd.angular.z = -rot_speed if offset > 0 else rot_speed
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)

        self._stop()
        self.get_logger().warn('CENTER: Timeout')
        return None

    def approach(self, target_area=None, timeout=None):
        if target_area is None:
            target_area = CV_LIFEBOAT_AREA_CLOSE
        if timeout is None:
            timeout = CV_APPROACH_TIMEOUT
        self.get_logger().info(
            f'APPROACH: Driving toward lifeboat until area>{target_area:.0f}...')
        t0 = time.time()
        cmd = Twist()
        lost_count = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin_cv(0.05)

            front_min = self._get_front_min()
            if front_min < LIDAR_FRONT_STOP:
                self._stop()
                self.get_logger().warn(
                    f'APPROACH: Front obstacle at {front_min:.2f}m — stopping')
                return self._get_best_detection('lifeboat')

            det = self._get_best_detection('lifeboat')
            if det is None:
                lost_count += 1
                if lost_count > 30:
                    self._stop()
                    self.get_logger().warn('APPROACH: Lost lifeboat')
                    return None
                cmd.linear.x = CV_LINEAR * 0.3
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue
            lost_count = 0

            if det['area'] >= target_area:
                self._stop()
                self.get_logger().info(
                    f'APPROACH: Lifeboat reached! area={det["area"]:.0f} >= {target_area:.0f}')
                return det

            offset = det['center'][0] - CV_CENTER_X
            steer = 0.0
            if abs(offset) > CV_CENTER_TOL * 0.5:
                steer = -offset / CAM_WIDTH * 1.0
                steer = max(-CV_ROTATE, min(CV_ROTATE, steer))

            speed = CV_LINEAR
            if front_min < LIDAR_SLOW_DIST:
                speed *= max(0.3, front_min / LIDAR_SLOW_DIST)

            cmd.linear.x = speed
            cmd.angular.z = steer
            self.cmd_pub.publish(cmd)

            elapsed = time.time() - t0
            if int(elapsed * 10) % 20 == 0:
                self.get_logger().info(
                    f'    area={det["area"]:.0f} offset={offset:.0f} '
                    f'speed={speed:.2f} steer={steer:.2f} front={front_min:.2f}')

            time.sleep(DT)

        self._stop()
        self.get_logger().warn('APPROACH: Timeout')
        return None

    def drive_until_lost(self, timeout=20.0):
        """Drive forward keeping the lifeboat centered until it disappears
        from the frame entirely."""
        self.get_logger().info('DRIVE: Moving forward until lifeboat leaves frame...')
        t0 = time.time()
        cmd = Twist()
        lost_count = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin_cv(0.05)

            front_min = self._get_front_min()
            if front_min < LIDAR_FRONT_STOP:
                self._stop()
                self.get_logger().warn(
                    f'DRIVE: Front obstacle at {front_min:.2f}m — stopping')
                return True  # close enough, obstacle stopped us

            det = self._get_best_detection('lifeboat')
            if det is None:
                lost_count += 1
                if lost_count > 45:  # lost for ~1.5s — gone from frame + 1s extra
                    self._stop()
                    self.get_logger().info(
                        'DRIVE: Lifeboat left the frame!')
                    return True
                # Keep driving straight
                cmd.linear.x = CV_LINEAR
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue
            lost_count = 0

            # Steer to keep it centered while driving forward
            offset = det['center'][0] - CV_CENTER_X
            steer = 0.0
            if abs(offset) > CV_CENTER_TOL * 0.5:
                steer = -offset / CAM_WIDTH * 1.0
                steer = max(-CV_ROTATE, min(CV_ROTATE, steer))

            speed = CV_LINEAR
            if front_min < LIDAR_SLOW_DIST:
                speed *= max(0.3, front_min / LIDAR_SLOW_DIST)

            cmd.linear.x = speed
            cmd.angular.z = steer
            self.cmd_pub.publish(cmd)

            elapsed = time.time() - t0
            if int(elapsed * 10) % 20 == 0:
                self.get_logger().info(
                    f'    area={det["area"]:.0f} offset={offset:.0f} '
                    f'front={front_min:.2f}')
            time.sleep(DT)

        self._stop()
        self.get_logger().warn('DRIVE: Timeout — lifeboat still in frame')
        return False

    def rotate_180_ccw(self):
        """Rotate counterclockwise exactly 180 degrees."""
        self.get_logger().info('ROTATE 180 CCW: Starting...')
        total_rotated = 0.0
        last_yaw = self._get_yaw()
        cmd = Twist()
        step = 0

        while rclpy.ok() and total_rotated < math.pi - 0.05:
            rclpy.spin_once(self, timeout_sec=0.05)
            yaw = self._get_yaw()
            if yaw is not None and last_yaw is not None:
                d_yaw = self._normalize_angle(yaw - last_yaw)
                if d_yaw > 0:  # only count CCW progress
                    total_rotated += d_yaw
                last_yaw = yaw

            cmd.angular.z = ROTATE_SPEED
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)

            step += 1
            if step % 30 == 0:
                self.get_logger().info(
                    f'    ROTATE: {math.degrees(total_rotated):.0f}/180 deg')
            time.sleep(DT)

        self._stop()
        self.get_logger().info(
            f'ROTATE 180 CCW: Done ({math.degrees(total_rotated):.0f} deg)')
        time.sleep(0.2)

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
    node = CVLifeboatTest()

    if not node.wait_for_cv():
        node.destroy_node()
        rclpy.shutdown()
        return

    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info('\n========== LIFEBOAT APPROACH TEST ==========')

    # Step 1: Find lifeboat (full 360° scan)
    node.set_status('Step 1: Scanning 360 for lifeboat')
    det = node.scan_for()
    if det is None:
        node.get_logger().error('Lifeboat not found after full rotation — test failed')
        node._stop()
        node.destroy_node()
        rclpy.shutdown()
        return

    # Step 2: Center on it
    node.set_status('Step 2: Centering on lifeboat')
    det = node.center_on()
    if det is None:
        node.get_logger().error('Could not center on lifeboat — test failed')
        node._stop()
        node.destroy_node()
        rclpy.shutdown()
        return

    # Step 3: Drive forward until lifeboat leaves the frame
    node.set_status('Step 3: Driving until lifeboat out of frame')
    result = node.drive_until_lost()
    if result:
        node.get_logger().info('\nLifeboat has left the frame!')
    else:
        node.get_logger().warn('\nDrive timed out — continuing anyway')

    # Step 4: Rotate 180° counterclockwise
    node.set_status('Step 4: Rotating 180 CCW')
    node.rotate_180_ccw()
    node.get_logger().info('180° rotation complete.')

    node.get_logger().info('========== TEST COMPLETE ==========')
    node.set_status('Test complete')
    node._stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
