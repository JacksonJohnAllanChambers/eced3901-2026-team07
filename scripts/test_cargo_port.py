#!/usr/bin/env python3
"""
==========================================================================
Standalone Cargo Port Interaction Test
==========================================================================
Tests the full cargo drop-off + pickup sequence at the port using the
CV detection pipeline (/cv/detections) and direct cmd_vel control.

Run modes:
  python3 test_cargo_port.py                 # full sequence (drop + pickup)
  python3 test_cargo_port.py --drop-only     # just drop-off
  python3 test_cargo_port.py --pickup-only   # just pickup

Prerequisites:
  1. ros2 launch dalmotor robot.launch.py
  2. ros2 launch eced3901 arena_real_amcl.launch.py
  3. ros2 launch dalibot_cv dalibot_cv_launch.py
"""

import argparse
import json
import math
import time

import serial

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
from std_msgs.msg import String


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

# Camera
CAM_WIDTH            = 1280
CAM_HEIGHT           = 720
CV_CENTER_X          = 640         # horizontal center of frame
CV_CENTER_TOL        = 80          # px — "centered" tolerance
CV_CENTER_TIGHT      = 30          # px — fine-align tolerance
CV_MIN_CONFIDENCE    = 0.35

# Cargo thresholds
CV_CARGO_AREA_CLOSE  = 35000       # area px² — close enough for drop/pickup
CARGO_SIDEWAYS_AR    = 1.2         # aspect ratio > this = sideways
PICKUP_BBOX_BOTTOM_Y = 600         # bbox bottom y — stop approaching

# Speeds
CV_LINEAR            = 0.05        # m/s forward during CV approach
CV_ROTATE            = 0.15        # rad/s rotation during CV scan
CV_ROTATE_PRECISION  = 0.10        # m/s even slower for tiny 1deg tweaks
REVERSE_SPEED        = 0.08

# Lidar
LIDAR_FRONT_STOP     = 0.12        # m — emergency stop
LIDAR_SLOW_DIST      = 0.50        # m — start slowing

# Timing
CARGO_DROP_PAUSE     = 3.0         # seconds to hold for drop
CARGO_PICKUP_PAUSE   = 5.0         # seconds to hold for pickup
CV_SCAN_TIMEOUT      = 15.0
CV_CENTER_TIMEOUT    = 10.0
CV_APPROACH_TIMEOUT  = 25.0
CV_PRECISION_SETTLE  = 0.6         #s to wait for CV/IMU to stabilize

# Go-around maneuver
GOAROUND_LATERAL     = 0.228       # m sideways offset (0.75 ft)
GOAROUND_FORWARD     = 0.228       # m forward past cargo (0.75 ft)

DT = 1.0 / 30.0

# Cargo USART
CARGO_PORT  = '/dev/ttyUSB4'   # FT232R connected to cargo ATmega328P
CARGO_BAUD  = 9600
CARGO_CMD_DROP   = b'2'
CARGO_CMD_PICKUP = b'1'


# ═══════════════════════════════════════════════════════════════
# CargoPortNode
# ═══════════════════════════════════════════════════════════════

class CargoPortNode(Node):
    def __init__(self):
        super().__init__('cargo_port_test')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/cv/nav_status', 10)

        # Subscribers
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

        # Cargo USART
        try:
            self._cargo_serial = serial.Serial(CARGO_PORT, CARGO_BAUD, timeout=1.0)
            self.get_logger().info(f'Cargo serial open: {CARGO_PORT} @ {CARGO_BAUD}')
        except Exception as e:
            self._cargo_serial = None
            self.get_logger().error(f'Cargo serial FAILED to open: {e}')

        self.get_logger().info('CargoPortNode ready')

    # ─── Callbacks ──────────────────────────────────────────────

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

    # ─── Status ─────────────────────────────────────────────────

    def set_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(f'STATUS: {text}')

    # ─── CV helpers ─────────────────────────────────────────────

    def _spin(self, duration=0.15):
        t0 = time.time()
        while time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _get_best_cargo(self):
        """Get highest-confidence cargo detection, or None.

        Filters:
          - Accepts 'cargo' or 'lifeboat' (lifeboat is often misclassified cargo)
          - Rejects vertical detections (aspect_ratio < 0.8 = taller than wide = invalid)
          - Rejects below minimum confidence
        """
        if time.time() - self.last_det_time > 1.0:
            return None
        best = None
        for d in self.last_detections:
            label = d.get('label')
            if label not in ('cargo', 'lifeboat'):
                continue
            if d.get('confidence', 0) < CV_MIN_CONFIDENCE:
                continue
            # Reject vertical detections — valid cargo is wider than tall
            ar = d.get('aspect_ratio', 1.0)
            if ar < 0.8:
                continue
            if best is None or d['confidence'] > best['confidence']:
                best = d
        return best

    def _detection_offset_x(self, det):
        """Horizontal offset from frame center (positive = object is right)."""
        return det['center'][0] - CV_CENTER_X

    def _detection_bbox_bottom(self, det):
        bbox = det.get('bbox', [0, 0, 0, 0])
        return bbox[1] + bbox[3]

    # ─── Lidar helpers ──────────────────────────────────────────

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

    # ─── Movement primitives ───────────────────────────────────

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

    def rotate_by(self, angle_rad, speed=None):
        """Rotate by relative angle (positive = CCW / left)."""
        if speed is None:
            speed = CV_ROTATE
        start_yaw = self._get_imu_yaw()
        if start_yaw is None:
            self.get_logger().warn('No IMU — using timed rotation')
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
        """Drive at given speed for given duration with lidar safety."""
        cmd = Twist()
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)
            front = self._get_front_min()
            if speed > 0 and front < LIDAR_FRONT_STOP:
                self.get_logger().warn(
                    f'EMERGENCY STOP: front={front:.2f}m')
                break
            if speed < 0:
                rear = self._get_rear_min()
                if rear < LIDAR_FRONT_STOP:
                    self.get_logger().warn(
                        f'REVERSE STOP: rear={rear:.2f}m')
                    break
            cmd.linear.x = speed
            self.cmd_pub.publish(cmd)
            time.sleep(DT)
        self._stop()
        time.sleep(0.1)

    # ═══════════════════════════════════════════════════════════
    # CV-GUIDED CARGO BEHAVIORS
    # ═══════════════════════════════════════════════════════════

    def scan_for_cargo(self, timeout=None):
        """Slowly rotate scanning for cargo. Returns det or None."""
        if timeout is None:
            timeout = CV_SCAN_TIMEOUT
        self.set_status('SCAN: Looking for cargo...')
        t0 = time.time()
        cmd = Twist()
        total_rotated = 0.0
        last_yaw = None

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.1)
            det = self._get_best_cargo()
            if det is not None:
                ar = det.get('aspect_ratio', 1.0)
                orient = 'sideways' if ar > CARGO_SIDEWAYS_AR else 'endwise'
                self.get_logger().info(
                    f'SCAN: Found cargo! conf={det["confidence"]:.2f} '
                    f'area={det["area"]:.0f} AR={ar:.2f} [{orient}]')
                self._stop()
                return det

            cmd.angular.z = CV_ROTATE
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)

            yaw = self._get_imu_yaw()
            if yaw is not None:
                if last_yaw is not None:
                    d_yaw = abs(self._normalize_angle(yaw - last_yaw))
                    total_rotated += d_yaw
                last_yaw = yaw
                if total_rotated > 2 * math.pi + 0.5:
                    self.get_logger().warn('SCAN: Full rotation — cargo not found')
                    break
            time.sleep(DT)

        self._stop()
        self.get_logger().warn('SCAN: Timeout — cargo not found')
        return None

    def center_on_cargo(self, timeout=None):
        """Rotate to center cargo in camera frame. Returns last det or None."""
        if timeout is None:
            timeout = CV_CENTER_TIMEOUT
        self.set_status('CENTER: Centering on cargo...')
        t0 = time.time()
        cmd = Twist()
        lost = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.1)
            det = self._get_best_cargo()
            if det is None:
                lost += 1
                if lost < 15:
                    time.sleep(DT)
                    continue
                # Drift slowly looking for it
                cmd.angular.z = CV_ROTATE * 0.3
                cmd.linear.x = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue
            lost = 0

            offset = self._detection_offset_x(det)
            if abs(offset) < CV_CENTER_TOL:
                self._stop()
                self.get_logger().info(
                    f'CENTER: Done (offset={offset:.0f}px)')
                return det

            spd = min(CV_ROTATE, max(0.04, abs(offset) / CAM_WIDTH * 0.3))
            cmd.angular.z = -spd if offset > 0 else spd
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)

        self._stop()
        self.get_logger().warn('CENTER: Timeout')
        return None

    def fine_align(self, timeout=5.0):
        """Tight centering — 30px tolerance, 5 consecutive good frames."""
        self.set_status('ALIGN: Fine alignment...')
        t0 = time.time()
        cmd = Twist()
        good = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.1)
            det = self._get_best_cargo()
            if det is None:
                time.sleep(DT)
                continue

            offset = self._detection_offset_x(det)
            if abs(offset) < CV_CENTER_TIGHT:
                good += 1
                self._stop()
                if good >= 5:
                    self.get_logger().info(
                        f'ALIGN: Locked (offset={offset:.0f}px)')
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
        self.get_logger().warn('ALIGN: Timeout')
        return None

    def orthogonal_align(self, timeout=25.0):
        """Maneuver to intersect the extended orthogonal centerline of the cargo face."""
        self.set_status('ORTHOGONAL ALIGN...')
        t0 = time.time()
        
        # 1. Stop and explicitly center the cargo in the frame
        self.center_on_cargo()
        
        # Give CV a moment to compute stable bounding box and top-edge angle
        self._spin(0.5)
        
        # Read the absolute distance D from Lidar
        D = self._get_front_min()
        if D < LIDAR_FRONT_STOP or D > 3.0:
            self.get_logger().warn(f"ORTHO ALIGN: Invalid Lidar distance: {D:.2f}m")
            return False
            
        # Read the orientation_error_deg (theta) from /cv/cargo_align
        if not self._cargo_align or time.time() - self._cargo_align_time > 2.0:
            self.get_logger().error("ORTHO ALIGN: No `/cv/cargo_align` telemetry found!")
            return False
            
        theta = self._cargo_align.get('orientation_error_deg')
        if theta is None:
            self.get_logger().error("ORTHO ALIGN: Cargo orientation data unavailable!")
            return False
            
        # If absolute theta is very small (< 5.0 deg), we are already aligned
        if abs(theta) < 5.0:
            self.get_logger().info(f"ORTHO ALIGN: Cargo already straight-on ({theta:+.1f} deg).")
            return True
            
        self.get_logger().info(f"ORTHO ALIGN: Cargo angled {theta:+.1f} deg at {D:.2f}m.")
        
        # Math: L = D * tan(|theta|)
        L = D * math.tan(math.radians(abs(theta)))
        self.get_logger().info(f"ORTHO ALIGN: Calculated sidestep L = {L:.3f}m")
        
        # Execute open-loop sidestep maneuver
        if theta < 0:
            # Right side is further away -> move RIGHT
            self.get_logger().info("  -> Sidestepping RIGHT")
            self.rotate_by(-math.pi / 2)
            self.drive_timed(CV_LINEAR, L / CV_LINEAR)
            self.rotate_by(math.pi / 2 + math.radians(abs(theta)))
        else:
            # Left side is further away -> move LEFT
            self.get_logger().info("  -> Sidestepping LEFT")
            self.rotate_by(math.pi / 2)
            self.drive_timed(CV_LINEAR, L / CV_LINEAR)
            self.rotate_by(-math.pi / 2 - math.radians(abs(theta)))
            
        self.get_logger().info("ORTHO ALIGN: Maneuver complete. Re-centering...")
        self.center_on_cargo()
        return True

    def approach_cargo(self, timeout=None):
        """Drive toward cargo keeping centered. Stop when bbox bottom
        reaches PICKUP_BBOX_BOTTOM_Y or area exceeds CV_CARGO_AREA_CLOSE."""
        if timeout is None:
            timeout = CV_APPROACH_TIMEOUT
        self.set_status('APPROACH: Driving toward cargo...')
        t0 = time.time()
        cmd = Twist()
        lost_count = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.05)

            front_min = self._get_front_min()
            if front_min < LIDAR_FRONT_STOP:
                self._stop()
                self.get_logger().info(
                    f'APPROACH: Lidar stop ({front_min:.2f}m)')
                return True

            det = self._get_best_cargo()
            if det is None:
                lost_count += 1
                if lost_count > 45:  # ~1.5s
                    self._stop()
                    self.get_logger().info(
                        'APPROACH: Cargo left frame — inching forward to engage')
                    self.drive_timed(CV_LINEAR * 0.4, 1.5)
                    return True
                cmd.linear.x = CV_LINEAR * 0.3
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue
            lost_count = 0

            # Check if close enough
            if det['area'] >= CV_CARGO_AREA_CLOSE:
                self._stop()
                self.get_logger().info(
                    f'APPROACH: Area threshold reached ({det["area"]:.0f})')
                return True

            bbox_bot = self._detection_bbox_bottom(det)
            if bbox_bot >= PICKUP_BBOX_BOTTOM_Y:
                self._stop()
                self.get_logger().info(
                    f'APPROACH: Frame bottom reached (bbox_bot={bbox_bot})')
                # Nudge forward to engage
                nudge = Twist()
                nudge.linear.x = CV_LINEAR * 0.5
                nt0 = time.time()
                while time.time() - nt0 < 0.8:
                    rclpy.spin_once(self, timeout_sec=0.05)
                    self.cmd_pub.publish(nudge)
                    time.sleep(DT)
                self._stop()
                return True

            offset = self._detection_offset_x(det)

            if int((time.time() - t0) * 10) % 20 == 0:
                self.get_logger().info(
                    f'    APPROACH: area={det["area"]:.0f} '
                    f'bbox_bot={bbox_bot} offset={offset:.0f} '
                    f'front={front_min:.2f}')

            # Steering
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
        self.get_logger().warn('APPROACH: Timeout')
        return False

    def _send_cargo_cmd(self, cmd: bytes, label: str):
        """Send a single-byte command to the cargo USART controller."""
        if self._cargo_serial is None:
            self.get_logger().error(f'Cargo serial not open — cannot send {label}')
            return
        try:
            self._cargo_serial.write(cmd)
            self._cargo_serial.flush()
            self.get_logger().info(f'Cargo CMD sent: {label} ({cmd!r})')
        except Exception as e:
            self.get_logger().error(f'Cargo serial write failed: {e}')

    def cargo_drop(self):
        self._send_cargo_cmd(CARGO_CMD_DROP, 'DROP')

    def cargo_pickup(self):
        self._send_cargo_cmd(CARGO_CMD_PICKUP, 'PICKUP')

    def reverse_out(self, duration=3.0):
        """Reverse with rear lidar safety."""
        self.set_status(f'REVERSE: Backing up {duration:.1f}s...')
        cmd = Twist()
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)
            rear = self._get_rear_min()
            if rear < LIDAR_FRONT_STOP:
                self.get_logger().warn(f'REVERSE: Rear obstacle at {rear:.2f}m')
                break
            cmd.linear.x = -REVERSE_SPEED
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)
        self._stop()
        time.sleep(0.2)

    # ═══════════════════════════════════════════════════════════
    # FULL PORT SEQUENCES
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

            det = self._get_best_cargo()
            if det is None:
                lost_count += 1
                if lost_count > 15: # ~0.5s out of frame
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

    def _backup_to_reacquire(self, dist_m=0.04):
        """Back up dist_m and wait for cargo to re-enter frame. Returns True if reacquired."""
        self.get_logger().info(f'PRECISION ALIGN: Backing up {dist_m*100:.0f}cm to re-acquire cargo...')
        self.drive_timed(-CV_LINEAR * 0.5, dist_m / (CV_LINEAR * 0.5))
        # Wait up to 3s for cargo to come back
        t_wait = time.time()
        while rclpy.ok() and time.time() - t_wait < 3.0:
            self._spin(0.1)
            c = self._cargo_align.get('cargo_detected', False)
            if c:
                self.get_logger().info('PRECISION ALIGN: Cargo reacquired after backup.')
                return True
        return False

    def precision_align(self, timeout=60.0):
        """Blended-arc alignment: one proportional step per fresh CV measurement.

        Each step applies:
            angular_z = -(K_YAW * yaw_err_deg + K_LATERAL * offset_x_px)
            linear_x  = STEP_SPEED  (small constant forward to make arc effective)

        After each step the loop waits for a genuinely fresh frame before
        re-measuring, eliminating the stale-data oscillation problem.
        """
        K_YAW      = 0.015  # rad/s per degree  (10° → 0.15, capped to MAX_ANG)
        K_LATERAL  = 0.0003 # rad/s per pixel   (100px → 0.03)
        STEP_SPEED = 0.015  # m/s forward during arc step
        STEP_DUR   = 0.4    # s per step
        MAX_ANG    = CV_ROTATE_PRECISION  # 0.10 rad/s hard cap
        YAW_TOL    = 2.5    # deg
        LAT_TOL    = 25     # px
        STABLE_REQ = 3      # consecutive good readings to declare success

        self.set_status('PRECISION ALIGN...')
        t0 = time.time()
        stable_count = 0
        lost_count = 0
        recovery_attempts = 0

        while rclpy.ok() and time.time() - t0 < timeout:

            # ── Wait for a genuinely fresh frame ──────────────────────
            prior_time = self._cargo_align_time
            flush_deadline = time.time() + 3.0
            while rclpy.ok() and time.time() < flush_deadline:
                self._spin(0.1)
                if self._cargo_align_time > prior_time + 0.15:
                    break

            if not self._cargo_align or time.time() - self._cargo_align_time > 1.5:
                self.get_logger().info('PRECISION ALIGN: Waiting for cargo_align data...')
                continue

            # ── Cargo lost recovery ───────────────────────────────────
            if not self._cargo_align.get('cargo_detected', False):
                lost_count += 1
                self.get_logger().info(f'PRECISION ALIGN: Cargo lost (frame {lost_count})')
                if lost_count < 8:
                    continue
                if lost_count < 20:
                    if recovery_attempts == 0 or lost_count % 8 == 0:
                        recovery_attempts += 1
                        if not self._backup_to_reacquire(0.04):
                            self.get_logger().warn('PRECISION ALIGN: Not reacquired after 4cm backup')
                    continue
                # Extended loss — larger backup + left/right sweep
                recovery_attempts += 1
                self.get_logger().warn(
                    f'PRECISION ALIGN: Extended loss — 10cm backup + sweep (attempt {recovery_attempts})')
                self.drive_timed(-CV_LINEAR * 0.5, 0.10 / (CV_LINEAR * 0.5))
                for direction in (1, -1, 1):
                    sweep_cmd = Twist()
                    sweep_cmd.angular.z = CV_ROTATE_PRECISION * direction
                    t_sweep = time.time()
                    while rclpy.ok() and time.time() - t_sweep < 0.8:
                        self._spin(0.05)
                        if self._cargo_align.get('cargo_detected', False):
                            self._stop()
                            self.get_logger().info('PRECISION ALIGN: Cargo found during sweep.')
                            break
                        self.cmd_pub.publish(sweep_cmd)
                        time.sleep(DT)
                    else:
                        self._stop()
                        continue
                    break
                lost_count = 0
                continue

            lost_count = 0
            recovery_attempts = 0

            # ── Read errors ───────────────────────────────────────────
            yaw_err  = self._cargo_align.get('orientation_error_deg')
            offset_x = self._cargo_align.get('offset_x')
            if yaw_err is None or offset_x is None:
                continue

            # ── Success check ─────────────────────────────────────────
            if abs(yaw_err) < YAW_TOL and abs(offset_x) < LAT_TOL:
                stable_count += 1
                self.get_logger().info(
                    f'PRECISION ALIGN: stable {stable_count}/{STABLE_REQ} '
                    f'(yaw={yaw_err:+.1f}°  off={offset_x:+.0f}px)')
                if stable_count >= STABLE_REQ:
                    self.get_logger().info('PRECISION ALIGN: Locked!')
                    return True
                continue
            stable_count = 0

            # ── Blended arc step ──────────────────────────────────────
            # Both errors drive the same sign: positive error → turn right (negative angular_z)
            raw_ang = -(K_YAW * yaw_err + K_LATERAL * offset_x)
            ang = max(-MAX_ANG, min(MAX_ANG, raw_ang))

            self.get_logger().info(
                f'PRECISION ALIGN: yaw={yaw_err:+.1f}°  off={offset_x:+.0f}px'
                f'  → ω={ang:+.3f} rad/s  step={STEP_DUR}s')

            cmd = Twist()
            cmd.angular.z = ang
            cmd.linear.x  = STEP_SPEED
            t_step = time.time()
            while rclpy.ok() and time.time() - t_step < STEP_DUR:
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
            self._stop()

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

        # Save to file
        with open('fine_approach_dist.txt', 'a') as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] Manual fine approach: {total_dist:.3f}m\n")
        self.get_logger().info(f'Saved manual fine approach distance: {total_dist:.3f}m to fine_approach_dist.txt')


    def do_port_sequence(self):
        """Full new cargo port interaction sequence as requested."""
        self.set_status('=== PORT SEQUENCE START ===')

        # 1. Scan for the cargo
        self.get_logger().info('--- Step 1: Scan for cargo ---')
        det = self.scan_for_cargo()
        if det is None:
            self.get_logger().error('No cargo found — aborting')
            return False

        # 2. Center and Fine align
        self.get_logger().info('--- Step 2: Center & Align ---')
        self.center_on_cargo()
        self.fine_align()

        # 2.5 Orthogonal Align
        self.get_logger().info('--- Step 2.5: Orthogonal Maneuver ---')
        self.orthogonal_align()

        # 3. Approach to center of frame (or close)
        self.get_logger().info('--- Step 3: Initial Approach ---')
        self.approach_cargo()
        self.get_logger().info('  Nudging closer before drop...')
        self.drive_timed(CV_LINEAR, 1.5)

        # 4. Check orientation
        # We need a fresh detection to check aspect ratio
        self._spin(0.5)
        det = self._get_best_cargo()
        if det is None:
            self.get_logger().error('Lost cargo after approach!')
            return False
        
        ar = det.get('aspect_ratio', 1.0)
        is_sideways = (ar > CARGO_SIDEWAYS_AR)
        orient_str = 'SIDEWAYS' if is_sideways else 'END-ON'
        self.get_logger().info(f'--- Step 4: Orientation Check --- AR={ar:.2f} -> {orient_str}')

        if not is_sideways:
            # 5. Go-around maneuver (hardcoded for left_open)
            self.get_logger().info('--- Step 5: End-on Go-Around Maneuver ---')
            
            # Rotate 90 right
            self.get_logger().info('  Rotate 90° Right')
            self.rotate_by(-math.pi / 2)
            
            # Go forward 0.75ft
            self.get_logger().info('  Drive Forward 0.75ft')
            self.drive_timed(CV_LINEAR, 0.228 / CV_LINEAR)
            
            # Rotate 90 left
            self.get_logger().info('  Rotate 90° Left')
            self.rotate_by(math.pi / 2)
            
            # Go forward 0.75ft
            self.get_logger().info('  Drive Forward 0.75ft')
            self.drive_timed(CV_LINEAR, 0.228 / CV_LINEAR)
            
            # Rotate 90 left again
            self.get_logger().info('  Rotate 90° Left')
            self.rotate_by(math.pi / 2)
            
            # Realign with the cargo
            self.get_logger().info('  Realigning with cargo...')
            self.scan_for_cargo()
            self.center_on_cargo()
            self.fine_align()
            self.approach_cargo()
            self.get_logger().info('  Nudging closer before drop...')
            self.drive_timed(CV_LINEAR, 1.5)

        # 6. Now side-by-side dropping
        self.get_logger().info('--- Step 6: Side-by-side Drop ---')
        self.get_logger().info('  Rotate 90° Left to parallel')
        self.rotate_by(math.pi / 2, speed=0.3)
        
        self.set_status('DROPPING CARGO')
        self._stop()
        self.cargo_drop()
        self.get_logger().info('>>> CARGO DROPPED <<<')
        time.sleep(1.0)  # hold for mechanism to actuate

        # 7. Rotate back to face target
        self.get_logger().info('--- Step 7: Rotate to face target ---')
        self.get_logger().info('  Rotate 90° Right to face target')
        self.rotate_by(-math.pi / 2, speed=0.3)

        # 7.5 Center on cargo
        self.get_logger().info('--- Step 7.5: Center on cargo ---')
        self.center_on_cargo()

        # 8. Drive in until cargo disappears from frame (contact)
        self.get_logger().info('--- Step 8: Approach and contact ---')
        self.approach_cargo_blind()

        # 8.2 Nudge forward a touch further to ensure full contact
        self.get_logger().info('--- Step 8.2: Final nudge ---')
        self.drive_timed(CV_LINEAR * 0.5, 0.08 / (CV_LINEAR * 0.5))
        self._stop()

        # 8.5 Pickup — pause first for magnets to drop and seat
        self.get_logger().info('--- Step 8.5: Pickup ---')
        self.set_status('PICKING UP CARGO')
        time.sleep(1.0)  # let magnets drop and seat against cargo
        self.cargo_pickup()
        self.get_logger().info('>>> CARGO PICKED UP <<<')
        time.sleep(1.5)  # hold while mechanism actuates

        # 9. Reverse out and verify cargo is gone
        self.get_logger().info('--- Step 9: Reverse out and verify ---')
        self.reverse_out(duration=2.5)
        self._spin(0.5)
        det = self._get_best_cargo()
        if det is not None:
            self.get_logger().warn('VERIFY: Cargo still detected — pickup may have failed!')
        else:
            self.get_logger().info('VERIFY: Cargo not detected — pickup confirmed.')

        self.set_status('PORT SEQUENCE COMPLETE')
        return True

    def wait_for_cv(self, timeout=10.0):
        """Wait until CV detection data arrives."""
        self.get_logger().info('Waiting for CV pipeline...')
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.last_det_time > 0:
                self.get_logger().info('CV pipeline active!')
                return True
        self.get_logger().warn('CV pipeline not detected — continuing')
        return False


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    rclpy.init()
    node = CargoPortNode()

    node.get_logger().info('=' * 50)
    node.get_logger().info('  CARGO PORT INTERACTION TEST')
    node.get_logger().info('=' * 50)

    # Wait for CV
    node.wait_for_cv()

    # Let subscribers populate
    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.1)

    node.get_logger().info('Mode: FULL SEQUENCE (left_open hardcoded)')

    # Run the full integrated port sequence
    success = node.do_port_sequence()
    if success:
        node.get_logger().info('>>> FULL PORT SEQUENCE COMPLETE <<<')
    else:
        node.get_logger().warn('Port sequence handled an error or aborted')

    node.get_logger().info('=' * 50)
    node.get_logger().info('  TEST COMPLETE')
    node.get_logger().info('=' * 50)
    node._stop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
