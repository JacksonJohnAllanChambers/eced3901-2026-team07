#!/usr/bin/env python3
"""
==========================================================================
Enhanced Open Water Navigation — Nav2 + CV Cargo + Lifeboat Detection
==========================================================================
Fork of simple_open_water.py that adds:
  - Cargo drop-off and pickup at the target port (via CV)
  - Lifeboat detection and rescue (via CV scan + approach)

Uses Nav2 BasicNavigator for route navigation and direct cmd_vel for
CV-guided precision operations (cargo approach, lifeboat rescue).

Mission phases:
  1. INIT      — Set AMCL pose, wait for Nav2 + CV pipeline
  2. TO PORT   — goToPose(port), preempt near goal
  3. CARGO     — Drop starting cargo, go-around, pick up target cargo
  4. SCAN      — 360° CV scan for lifeboat at port area
  5. HOME      — goToPose(home) with periodic lifeboat checks
  6. LIFEBOAT  — If detected, approach + pause for rescue
  7. DONE      — Report final results

Usage:
  python3 open_water_full.py                          # default: left_open
  python3 open_water_full.py --lane right_open        # right lane
  python3 open_water_full.py --lane left_open --skip-cargo  # skip cargo ops

Prerequisites:
  1. ros2 launch dalmotor robot.launch.py
  2. ros2 launch eced3901 arena_real_amcl.launch.py
  3. ros2 launch dalibot_cv dalibot_cv_launch.py
"""

import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import String
import numpy as np


# ═══════════════════════════════════════════════════════════════
# Lane coordinates (SLAM frame)
# ═══════════════════════════════════════════════════════════════

OPEN_WATER_COORDS = {
    'left_open': {
        'x': 1.523,
        'y': 0.305,
        'goal_y': 3.67,
        'home_yaw': -90.0,
    },
    'right_open': {
        'x': 2.742,
        'y': 0.305,
        'goal_y': 3.67,
        'home_yaw': 90.0,
    },
}

# ── CV constants ──
CAM_WIDTH            = 1280
CV_CENTER_X          = 640
CV_CENTER_TOL        = 80
CV_CENTER_TIGHT      = 30
CV_MIN_CONFIDENCE    = 0.35
CV_CARGO_AREA_CLOSE  = 35000
CARGO_SIDEWAYS_AR    = 1.2
PICKUP_BBOX_BOTTOM_Y = 600

# ── CV approach speeds ──
CV_LINEAR            = 0.10
CV_ROTATE            = 0.20
REVERSE_SPEED        = 0.10

# ── Lidar ──
LIDAR_FRONT_STOP     = 0.12
LIDAR_SLOW_DIST      = 0.50

# ── Lifeboat ──
LIFEBOAT_MIN_AREA    = 300
LIFEBOAT_MAX_AREA    = 12000
LIFEBOAT_PAUSE_SEC   = 10.0
CV_LIFEBOAT_AREA_CLOSE = 8000
LIFEBOAT_FRAME_BOT_Y  = 620

# ── Timing ──
CARGO_DROP_PAUSE     = 3.0
CARGO_PICKUP_PAUSE   = 5.0
CV_SCAN_TIMEOUT      = 15.0
CV_CENTER_TIMEOUT    = 10.0
CV_APPROACH_TIMEOUT  = 25.0

# ── Go-around ──
GOAROUND_LATERAL     = 0.228       # m sideways offset
GOAROUND_FORWARD     = 0.228       # m forward past cargo

DT = 1.0 / 30.0


# ═══════════════════════════════════════════════════════════════
# Helper: create Nav2 pose
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


# ═══════════════════════════════════════════════════════════════
# CV + Movement Node (runs alongside BasicNavigator)
# ═══════════════════════════════════════════════════════════════

class CVHelper(Node):
    """Lightweight node for CV detection + direct cmd_vel control.
    Used for cargo interaction and lifeboat rescue — NOT for route nav."""

    def __init__(self):
        super().__init__('cv_helper')

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/cv/nav_status', 10)

        self.last_detections = []
        self.last_det_time = 0.0
        self.create_subscription(
            String, '/cv/detections', self._det_cb, 10)

        self.last_scan = None
        self.create_subscription(
            LaserScan, '/scan', self._scan_cb, 10)

        self.last_imu = None
        self.create_subscription(
            Imu, '/imu/imu', self._imu_cb, 10)

        self._cargo_align = {}
        self._cargo_align_time = 0.0
        self.create_subscription(
            String, '/cv/cargo_align', self._cargo_align_cb, 10)

    def _det_cb(self, msg):
        try:
            self.last_detections = json.loads(msg.data)
            self.last_det_time = time.time()
        except json.JSONDecodeError:
            self.last_detections = []

    def _scan_cb(self, msg):
        self.last_scan = msg

    def _imu_cb(self, msg):
        self.last_imu = msg

    def _cargo_align_cb(self, msg):
        try:
            self._cargo_align = json.loads(msg.data)
            self._cargo_align_time = time.time()
        except json.JSONDecodeError:
            pass

    def set_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(f'STATUS: {text}')

    def _spin(self, duration=0.15):
        t0 = time.time()
        while time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _get_imu_yaw(self):
        if self.last_imu is None:
            return None
        q = self.last_imu.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    @staticmethod
    def _normalize_angle(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

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

    def _get_rear_min(self):
        return self._lidar_sector_min(160, 200)

    # ─── Detection helpers ──────────────────────────────────────

    def _get_best(self, label):
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

    def _offset_x(self, det):
        return det['center'][0] - CV_CENTER_X

    def _bbox_bottom(self, det):
        bbox = det.get('bbox', [0, 0, 0, 0])
        return bbox[1] + bbox[3]

    def _validate_lifeboat(self, det):
        if det is None:
            return False
        area = det.get('area', 0)
        if area < LIFEBOAT_MIN_AREA or area > LIFEBOAT_MAX_AREA:
            return False
        return True

    # ─── Movement ──────────────────────────────────────────────

    def rotate_by(self, angle_rad, speed=None):
        if speed is None:
            speed = CV_ROTATE
        start_yaw = self._get_imu_yaw()
        if start_yaw is None:
            cmd = Twist()
            duration = abs(angle_rad) / speed
            cmd.angular.z = speed if angle_rad > 0 else -speed
            t0 = time.time()
            while time.time() - t0 < duration:
                self.cmd_pub.publish(cmd)
                rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(DT)
            self._stop()
            return

        target_yaw = self._normalize_angle(start_yaw + angle_rad)
        cmd = Twist()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            yaw = self._get_imu_yaw()
            if yaw is None:
                time.sleep(DT)
                continue
            err = self._normalize_angle(target_yaw - yaw)
            if abs(err) < 0.05:
                break
            spd = min(speed, max(0.06, abs(err) * 0.8))
            cmd.angular.z = spd if err > 0 else -spd
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)
        self._stop()
        time.sleep(0.1)

    def drive_timed(self, speed, duration):
        cmd = Twist()
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)
            front = self._get_front_min()
            if speed > 0 and front < LIDAR_FRONT_STOP:
                break
            if speed < 0:
                rear = self._get_rear_min()
                if rear < LIDAR_FRONT_STOP:
                    break
            cmd.linear.x = speed
            self.cmd_pub.publish(cmd)
            time.sleep(DT)
        self._stop()
        time.sleep(0.1)

    # ═══════════════════════════════════════════════════════════
    # CV SCAN / CENTER / APPROACH (works for both cargo and lifeboat)
    # ═══════════════════════════════════════════════════════════

    def cv_scan(self, label, timeout=None):
        """360° scan looking for label. Returns det or None."""
        if timeout is None:
            timeout = CV_SCAN_TIMEOUT
        self.get_logger().info(f'CV SCAN: Looking for {label}...')
        t0 = time.time()
        cmd = Twist()
        total_rotated = 0.0
        last_yaw = None

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.1)
            det = self._get_best(label)
            if det is not None:
                if label == 'lifeboat' and not self._validate_lifeboat(det):
                    time.sleep(DT)
                    continue
                self.get_logger().info(
                    f'CV SCAN: Found {label}! conf={det["confidence"]:.2f} '
                    f'area={det["area"]:.0f}')
                self._stop()
                return det

            cmd.angular.z = CV_ROTATE
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)

            yaw = self._get_imu_yaw()
            if yaw is not None:
                if last_yaw is not None:
                    total_rotated += abs(self._normalize_angle(yaw - last_yaw))
                last_yaw = yaw
                if total_rotated > 2 * math.pi + 0.5:
                    break
            time.sleep(DT)

        self._stop()
        self.get_logger().warn(f'CV SCAN: {label} not found')
        return None

    def cv_scan_partial(self, label, angle_range=math.pi / 4, timeout=10.0):
        """±angle_range sweep. Returns det or None."""
        self.get_logger().info(
            f'CV PARTIAL: ±{math.degrees(angle_range):.0f}° for {label}')
        t0 = time.time()

        start_yaw = self._get_imu_yaw()
        if start_yaw is None:
            return None

        for direction in [1, -1]:
            target = self._normalize_angle(start_yaw + direction * angle_range)
            cmd = Twist()
            while rclpy.ok() and time.time() - t0 < timeout:
                self._spin(0.05)
                det = self._get_best(label)
                if det is not None:
                    if label == 'lifeboat' and not self._validate_lifeboat(det):
                        time.sleep(DT)
                        continue
                    self._stop()
                    return det

                yaw = self._get_imu_yaw()
                if yaw is None:
                    time.sleep(DT)
                    continue
                err = self._normalize_angle(target - yaw)
                if abs(err) < 0.05:
                    break
                spd = min(CV_ROTATE, max(0.06, abs(err) * 0.8))
                cmd.angular.z = spd if err > 0 else -spd
                cmd.linear.x = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
            self._stop()
            time.sleep(0.1)

        # Return to start heading
        self.rotate_by(self._normalize_angle(start_yaw - (self._get_imu_yaw() or start_yaw)))
        return None

    def cv_center(self, label, timeout=None):
        """Rotate to center detection in frame."""
        if timeout is None:
            timeout = CV_CENTER_TIMEOUT
        t0 = time.time()
        cmd = Twist()
        lost = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.1)
            det = self._get_best(label)
            if det is None:
                lost += 1
                if lost < 15:
                    time.sleep(DT)
                    continue
                cmd.angular.z = CV_ROTATE * 0.3
                cmd.linear.x = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue
            lost = 0

            offset = self._offset_x(det)
            if abs(offset) < CV_CENTER_TOL:
                self._stop()
                return det

            spd = min(CV_ROTATE, max(0.04, abs(offset) / CAM_WIDTH * 0.5))
            cmd.angular.z = -spd if offset > 0 else spd
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)

        self._stop()
        return None

    def cv_fine_align(self, label='cargo', timeout=5.0):
        """Tight centering — 30px, 5 consecutive good frames."""
        t0 = time.time()
        cmd = Twist()
        good = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.1)
            det = self._get_best(label)
            if det is None:
                time.sleep(DT)
                continue
            offset = self._offset_x(det)
            if abs(offset) < CV_CENTER_TIGHT:
                good += 1
                self._stop()
                if good >= 5:
                    return det
                time.sleep(DT)
                continue
            good = 0
            spd = min(0.08, max(0.03, abs(offset) / CAM_WIDTH * 0.3))
            cmd.angular.z = -spd if offset > 0 else spd
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)

        self._stop()
        return None

    def cv_approach(self, label, target_area, timeout=None):
        """Drive toward label keeping centered until area >= target_area."""
        if timeout is None:
            timeout = CV_APPROACH_TIMEOUT
        t0 = time.time()
        cmd = Twist()
        lost_count = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.05)

            front_min = self._get_front_min()
            if front_min < LIDAR_FRONT_STOP:
                self._stop()
                return self._get_best(label)

            det = self._get_best(label)
            if det is None:
                lost_count += 1
                if lost_count > 45:
                    self._stop()
                    return None
                cmd.linear.x = CV_LINEAR * 0.3
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue
            lost_count = 0

            if det['area'] >= target_area:
                self._stop()
                return det

            bbox_bot = self._bbox_bottom(det)
            if label == 'cargo' and bbox_bot >= PICKUP_BBOX_BOTTOM_Y:
                self._stop()
                # Nudge forward
                nudge = Twist()
                nudge.linear.x = CV_LINEAR * 0.5
                nt0 = time.time()
                while time.time() - nt0 < 0.8:
                    rclpy.spin_once(self, timeout_sec=0.05)
                    self.cmd_pub.publish(nudge)
                    time.sleep(DT)
                self._stop()
                return det

            if label == 'lifeboat' and bbox_bot >= LIFEBOAT_FRAME_BOT_Y:
                self._stop()
                return det

            offset = self._offset_x(det)
            steer = 0.0
            if abs(offset) > CV_CENTER_TOL * 0.5:
                steer = -offset / CAM_WIDTH * 0.8
                steer = max(-CV_ROTATE, min(CV_ROTATE, steer))

            speed = CV_LINEAR
            if front_min < LIDAR_SLOW_DIST:
                speed *= max(0.3, front_min / LIDAR_SLOW_DIST)

            cmd.linear.x = speed
            cmd.angular.z = steer
            self.cmd_pub.publish(cmd)
            time.sleep(DT)

        self._stop()
        return None

    def reverse_out(self, duration=3.0):
        cmd = Twist()
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)
            rear = self._get_rear_min()
            if rear < LIDAR_FRONT_STOP:
                break
            cmd.linear.x = -REVERSE_SPEED
            self.cmd_pub.publish(cmd)
            time.sleep(DT)
        self._stop()
        time.sleep(0.2)

    # ═══════════════════════════════════════════════════════════
    # HIGH-LEVEL SEQUENCES
    # ═══════════════════════════════════════════════════════════

    def approach_cargo_blind(self, timeout=15.0):
        """Drive forward until cargo leaves the frame (gets too close)."""
        self.set_status('BLIND APPROACH...')
        t0 = time.time()
        cmd = Twist()
        lost_count = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.05)
            front_min = self._get_front_min()
            if front_min < LIDAR_FRONT_STOP:
                self._stop()
                self.get_logger().info(f'BLIND APPROACH: Lidar stop ({front_min:.2f}m)')
                return

            det = self._get_best('cargo')
            if det is None:
                lost_count += 1
                if lost_count > 15:
                    self._stop()
                    self.get_logger().info('BLIND APPROACH: Cargo out of frame — stopping.')
                    return
            else:
                lost_count = 0
                
            cmd.linear.x = CV_LINEAR * 0.5
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)

        self._stop()
        self.get_logger().warn('BLIND APPROACH: Timeout')

    def precision_align(self, timeout=45.0):
        """Uses /cv/cargo_align (washers) to achieve 0-degree yaw and 0-px lateral error."""
        self.set_status('PRECISION ALIGN...')
        t0 = time.time()
        stable_count = 0
        
        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.2)
            if not self._cargo_align or time.time() - self._cargo_align_time > 1.0:
                self.get_logger().info('PRECISION ALIGN: Waiting for /cv/cargo_align data...')
                continue
                
            c_det = self._cargo_align.get('cargo_detected', False)
            if not c_det:
                self.get_logger().info('PRECISION ALIGN: Cargo lost in align topic.')
                continue
                
            yaw_err = self._cargo_align.get('orientation_error_deg')
            offset_x = self._cargo_align.get('offset_x')
            
            # 1. Yaw Correction
            if yaw_err is not None and abs(yaw_err) > 1.5:
                stable_count = 0
                self.get_logger().info(f'PRECISION ALIGN: Fixing yaw {yaw_err:+.1f} deg')
                turn_rad = math.copysign(2.0 * math.pi / 180.0, -yaw_err) 
                self.rotate_by(turn_rad)
                time.sleep(0.5)
                continue
                
            # 2. Lateral Correction via Zig-Zag
            if offset_x is not None and abs(offset_x) > 15:
                stable_count = 0
                self.get_logger().info(f'PRECISION ALIGN: Fixing lateral offset {offset_x:+.0f} px')
                turn_rad = 15.0 * math.pi / 180.0
                direction = -1.0 if offset_x > 0 else 1.0
                
                self.rotate_by(direction * turn_rad)
                dist_m = max(0.01, min(0.05, abs(offset_x) * 0.0005))
                self.drive_timed(CV_LINEAR, dist_m / CV_LINEAR)
                self.rotate_by(-direction * turn_rad)
                time.sleep(0.5)
                continue
                
            # Both within tolerance
            stable_count += 1
            if stable_count >= 2:
                self.get_logger().info('PRECISION ALIGN: Successfully aligned!')
                return True
                
        self.get_logger().warn('PRECISION ALIGN: Timeout')
        return False

    def manual_fine_approach(self):
        """Keyboard control to inch forward/backward, logs total distance."""
        self.set_status('MANUAL APPROACH')
        print("\n" + "="*50)
        print("  MANUAL FINE APPROACH")
        print("  Controls:")
        print("    'w' + ENTER : Forward 0.02m (2cm)")
        print("    's' + ENTER : Backward 0.02m (2cm)")
        print("    'q' + ENTER : Done (or just ENTER)")
        print("="*50)

        total_dist = 0.0
        step = 0.02 

        while rclpy.ok():
            cmd_in = input(f"Total: {total_dist:.2f}m. Action [w/s/q]: ").strip().lower()
            if cmd_in == 'w':
                self.drive_timed(CV_LINEAR * 0.5, step / (CV_LINEAR * 0.5))
                total_dist += step
            elif cmd_in == 's':
                self.drive_timed(-CV_LINEAR * 0.5, step / (CV_LINEAR * 0.5))
                total_dist -= step
            elif cmd_in in ['q', '']:
                break

        with open('fine_approach_dist.txt', 'a') as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] Manual fine approach: {total_dist:.3f}m\n")
        self.get_logger().info(f'Saved manual fine approach distance: {total_dist:.3f}m to fine_approach_dist.txt')

    def do_full_cargo_sequence(self):
        """Full new cargo port interaction sequence as requested."""
        self.set_status('=== PORT SEQUENCE START ===')

        self.get_logger().info('--- Step 1: Scan for cargo ---')
        det = self.cv_scan('cargo')
        if det is None:
            self.get_logger().error('No cargo found — aborting')
            return False

        self.get_logger().info('--- Step 2: Center & Align ---')
        self.cv_center('cargo')
        self.cv_fine_align('cargo')

        self.get_logger().info('--- Step 3: Initial Approach ---')
        self.cv_approach('cargo', CV_CARGO_AREA_CLOSE)

        self._spin(0.5)
        det = self._get_best('cargo')
        if det is None:
            self.get_logger().error('Lost cargo after approach!')
            return False
        
        ar = det.get('aspect_ratio', 1.0)
        is_sideways = (ar > CARGO_SIDEWAYS_AR)
        orient_str = 'SIDEWAYS' if is_sideways else 'END-ON'
        self.get_logger().info(f'--- Step 4: Orientation Check --- AR={ar:.2f} -> {orient_str}')

        if not is_sideways:
            self.get_logger().info('--- Step 5: End-on Go-Around Maneuver ---')
            self.get_logger().info('  Rotate 90° Right')
            self.rotate_by(-math.pi / 2)
            self.get_logger().info('  Drive Forward 0.75ft')
            self.drive_timed(CV_LINEAR, 0.228 / CV_LINEAR)
            self.get_logger().info('  Rotate 90° Left')
            self.rotate_by(math.pi / 2)
            self.get_logger().info('  Drive Forward 0.75ft')
            self.drive_timed(CV_LINEAR, 0.228 / CV_LINEAR)
            self.get_logger().info('  Rotate 90° Left')
            self.rotate_by(math.pi / 2)
            self.get_logger().info('  Realigning with cargo...')
            self.cv_scan('cargo')
            self.cv_center('cargo')
            self.cv_fine_align('cargo')
            self.cv_approach('cargo', CV_CARGO_AREA_CLOSE)

        self.get_logger().info('--- Step 6: Side-by-side Drop ---')
        self.get_logger().info('  Rotate 90° Right to parallel')
        self.rotate_by(-math.pi / 2)
        
        self.set_status('DROPPING CARGO — waiting for operator')
        self._stop()
        try:
            input('\n  ⏸️  Trigger cargo DROP, then press ENTER to continue... ')
        except (KeyboardInterrupt, EOFError):
            self.get_logger().warn('Operator cancelled drop')
            return False
        self.get_logger().info('>>> CARGO DROPPED <<<')

        self.get_logger().info('--- Step 7: Rotate to face target ---')
        self.get_logger().info('  Rotate 90° Left to face target')
        self.rotate_by(math.pi / 2)

        self.get_logger().info('--- Step 7.5: Precision Washer Align ---')
        self.precision_align()

        self.get_logger().info('--- Step 8: Blind approach ---')
        self.approach_cargo_blind()

        self.get_logger().info('--- Step 9: Manual fine approach ---')
        self.manual_fine_approach()

        self.set_status('PICKING UP CARGO — waiting for operator')
        self._stop()
        try:
            input('\n  ⏸️  Trigger cargo PICKUP, then press ENTER to continue... ')
        except (KeyboardInterrupt, EOFError):
            self.get_logger().warn('Operator cancelled pickup')
            return False
        self.get_logger().info('>>> CARGO PICKED UP <<<')

        self.set_status('PORT SEQUENCE COMPLETE')
        return True

    def do_lifeboat_rescue(self):
        """Center on lifeboat → approach → pause for rescue."""
        self.set_status('LIFEBOAT: Rescue starting')

        det = self.cv_center('lifeboat')
        if det is None:
            self.get_logger().warn('LIFEBOAT: Lost during centering')
            return False

        result = self.cv_approach('lifeboat', CV_LIFEBOAT_AREA_CLOSE)
        if result is None:
            self.get_logger().warn('LIFEBOAT: Approach failed')
            return False

        self.set_status(f'LIFEBOAT: RESCUE — pausing {LIFEBOAT_PAUSE_SEC:.0f}s')
        self._stop()
        # TODO: Send rescue mechanism command via UART if needed
        t0 = time.time()
        while time.time() - t0 < LIFEBOAT_PAUSE_SEC:
            rclpy.spin_once(self, timeout_sec=0.2)
            remaining = LIFEBOAT_PAUSE_SEC - (time.time() - t0)
            if int(remaining) % 3 == 0 and abs(remaining - int(remaining)) < 0.3:
                self.get_logger().info(
                    f'  Rescue in progress... {remaining:.0f}s')

        self.get_logger().info('>>> LIFEBOAT RESCUED <<<')
        self.set_status('LIFEBOAT: RESCUE COMPLETE')
        return True

    def wait_for_cv(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.last_det_time > 0:
                self.get_logger().info('CV pipeline active!')
                return True
        self.get_logger().warn('CV pipeline not detected — continuing')
        return False

    def quick_lifeboat_check(self):
        """Non-blocking: check if lifeboat is visible right now."""
        self._spin(0.3)
        det = self._get_best('lifeboat')
        if det is not None and self._validate_lifeboat(det):
            return det
        return None


# ═══════════════════════════════════════════════════════════════
# Main Mission
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Open Water Nav — Cargo + Lifeboat')
    parser.add_argument('--lane', type=str, default='left_open',
                        choices=['left_open', 'right_open'],
                        help='Which open water lane (default: left_open)')
    parser.add_argument('--skip-cargo', action='store_true',
                        help='Skip cargo operations at port')
    parser.add_argument('--skip-lifeboat', action='store_true',
                        help='Skip lifeboat scanning')
    args = parser.parse_args()

    rclpy.init()

    data = OPEN_WATER_COORDS[args.lane]

    # ─── Create Nav2 navigator ──────────────────────────────────
    nav = BasicNavigator()

    # ─── Create CV helper node ──────────────────────────────────
    cv = CVHelper()

    print(f'\n{"=" * 55}')
    print(f'  ENHANCED OPEN WATER NAV — {args.lane.upper()}')
    print(f'  Cargo: {"SKIP" if args.skip_cargo else "ON"}')
    print(f'  Lifeboat: {"SKIP" if args.skip_lifeboat else "ON"}')
    print(f'{"=" * 55}\n')

    lifeboat_rescued = False

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: INIT
    # ═══════════════════════════════════════════════════════════
    print(f'📍 Phase 1: Initializing AMCL for {args.lane.upper()}...')
    cv.set_status('Phase 1: INIT')

    init_pose = create_pose(nav, data['x'], data['y'], 90.0)
    nav.setInitialPose(init_pose)

    print('⏳ Waiting for Nav2/AMCL to stabilize...')
    nav.waitUntilNav2Active(localizer='amcl')
    print('✅ Nav2 Ready!')

    # Wait for CV pipeline
    cv.wait_for_cv()
    for _ in range(30):
        rclpy.spin_once(cv, timeout_sec=0.1)

    # ═══════════════════════════════════════════════════════════
    # PHASE 2: NAVIGATE TO PORT
    # ═══════════════════════════════════════════════════════════
    print(f'\n🚀 Phase 2: Navigating to port...')
    cv.set_status('Phase 2: TO PORT')

    goal_pose = create_pose(nav, data['x'], data['goal_y'], 90.0)
    nav.goToPose(goal_pose)

    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback:
            dist = feedback.distance_remaining
            print(f'  Distance to port: {dist:.2f} m', end='\r')

            # Quick lifeboat check while navigating (opportunistic)
            if not args.skip_lifeboat and not lifeboat_rescued:
                lb = cv.quick_lifeboat_check()
                if lb is not None:
                    print(f'\n  👀 Lifeboat spotted during transit! '
                          f'area={lb["area"]:.0f}')
                    # Don't interrupt — save for later

            if dist < 0.40:
                print('\n🎯 Port reached!')
                break

    # ═══════════════════════════════════════════════════════════
    # PHASE 3: CARGO OPERATIONS
    # ═══════════════════════════════════════════════════════════
    if not args.skip_cargo:
        print(f'\n📦 Phase 3: Cargo operations at port...')
        cv.set_status('Phase 3: CARGO')

        cargo_ok = cv.do_full_cargo_sequence()
        if cargo_ok:
            print('✅ Cargo sequence complete!')
        else:
            print('⚠️  Cargo sequence had issues')
    else:
        print('\n⏩ Phase 3: Cargo skipped')

    time.sleep(1.0)

    # ═══════════════════════════════════════════════════════════
    # PHASE 4: LIFEBOAT SCAN AT PORT
    # ═══════════════════════════════════════════════════════════
    if not args.skip_lifeboat and not lifeboat_rescued:
        print(f'\n🔍 Phase 4: Scanning for lifeboat at port area...')
        cv.set_status('Phase 4: LIFEBOAT SCAN')

        det = cv.cv_scan('lifeboat', timeout=12.0)
        if det is not None:
            print(f'  🚨 Lifeboat detected! area={det["area"]:.0f}')
            lifeboat_rescued = cv.do_lifeboat_rescue()
            if lifeboat_rescued:
                print('✅ Lifeboat rescued!')
            else:
                print('⚠️  Lifeboat rescue failed')
        else:
            print('  No lifeboat found at port — will scan on return')
    else:
        print('\n⏩ Phase 4: Lifeboat scan skipped')

    # ═══════════════════════════════════════════════════════════
    # PHASE 5: NAVIGATE HOME
    # ═══════════════════════════════════════════════════════════
    print(f'\n🏠 Phase 5: Navigating home (yaw={data["home_yaw"]}°)...')
    cv.set_status('Phase 5: HOME')

    home_pose = create_pose(nav, data['x'], data['y'], data['home_yaw'])
    nav.goToPose(home_pose)

    scan_interval = 3.0  # seconds between lifeboat checks
    last_scan_time = time.time()

    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback:
            dist = feedback.distance_remaining
            print(f'  Distance to home: {dist:.2f} m', end='\r')

        # Periodic lifeboat check during return
        if (not args.skip_lifeboat and not lifeboat_rescued
                and time.time() - last_scan_time > scan_interval):
            last_scan_time = time.time()
            lb = cv.quick_lifeboat_check()
            if lb is not None:
                print(f'\n  🚨 LIFEBOAT SPOTTED during return! '
                      f'area={lb["area"]:.0f} conf={lb["confidence"]:.2f}')
                # Preempt Nav2
                nav.cancelTask()
                time.sleep(0.5)

                lifeboat_rescued = cv.do_lifeboat_rescue()
                if lifeboat_rescued:
                    print('✅ Lifeboat rescued!')
                else:
                    print('⚠️  Rescue failed — resuming home')

                # Resume navigation home
                print('  Resuming navigation home...')
                home_pose = create_pose(
                    nav, data['x'], data['y'], data['home_yaw'])
                nav.goToPose(home_pose)

    # ═══════════════════════════════════════════════════════════
    # PHASE 6: FINAL LIFEBOAT SCAN AT HOME
    # ═══════════════════════════════════════════════════════════
    if not args.skip_lifeboat and not lifeboat_rescued:
        print(f'\n🔍 Phase 6: Final lifeboat scan at home...')
        cv.set_status('Phase 6: FINAL SCAN')

        det = cv.cv_scan('lifeboat', timeout=12.0)
        if det is not None:
            lifeboat_rescued = cv.do_lifeboat_rescue()
    else:
        print('\n⏩ Phase 6: Final scan skipped')

    # ═══════════════════════════════════════════════════════════
    # PHASE 7: DONE
    # ═══════════════════════════════════════════════════════════
    cv.set_status('COMPLETE')
    cv._stop()

    print(f'\n{"=" * 55}')
    print(f'  MISSION COMPLETE')
    print(f'  Cargo: {"✅" if not args.skip_cargo else "⏩ skipped"}')
    print(f'  Lifeboat: {"✅ rescued" if lifeboat_rescued else "❌ not found"}')
    print(f'{"=" * 55}\n')

    cv.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
