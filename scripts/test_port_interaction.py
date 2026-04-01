#!/usr/bin/env python3
"""
Test Cargo Pickup — Horizontal (Sideways) Side Only
=====================================================
Simple test: point robot at cargo's horizontal/sideways side,
center on it, drive forward until prongs engage washer, done.

Place robot facing the sideways side of cargo from ~0.3-1m away.

Requires:
  1. ros2 launch dalmotor robot.launch.py
  2. ros2 launch dalibot_cv dalibot_cv_launch.py
  3. ros2 run eced3901 test_port_interaction.py
"""

import json
import math
import subprocess
import time

try:
    import serial as _serial_mod
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

SERIAL_DEVICE = '/dev/ttyUSB0'
SERIAL_BAUD   = 9600
CMD_PICKUP    = b'\x01'
CMD_DROP      = b'\x02'

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String, Bool

# ── Speeds ──
CV_LINEAR     = 0.06
CV_ROTATE     = 0.15
REVERSE_SPEED = 0.08
TURN_SPEED    = 0.55   # rad/s for drop turn

# ── Drop ──
DROP_DEG   = 90.0   # degrees CW to turn before dropping
DROP_PAUSE = 3.0    # s after drop command

# ── Lidar ──
LIDAR_FRONT_STOP = 0.12
LIDAR_SLOW_DIST  = 0.40

# ── CV ──
CV_CENTER_X       = 640
CV_CENTER_TOL     = 60
CV_MIN_CONFIDENCE = 0.35
CV_SCAN_TIMEOUT   = 20.0
CAM_WIDTH         = 1280

# ── Pickup threshold ──
# bbox bottom Y at which prongs are at the washer.
# Increase if stopping too early, decrease if overshooting.
PICKUP_BBOX_BOTTOM_Y = 580

DT = 1.0 / 30.0


class CargoPickupTest(Node):
    def __init__(self):
        super().__init__('cargo_pickup_test')

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

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

        self._serial = None
        self._open_serial()

    def _odom_cb(self, msg):  self.last_odom = msg
    def _imu_cb(self, msg):   self.last_imu = msg
    def _scan_cb(self, msg):  self.last_scan = msg

    def _det_cb(self, msg):
        try:
            self.last_detections = json.loads(msg.data)
            self.last_det_time = time.time()
        except json.JSONDecodeError:
            self.last_detections = []

    # ─── Helpers ────────────────────────────────────────────────
    def _spin(self, duration=0.1):
        t0 = time.time()
        while time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)

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
        return self._lidar_sector_min(165, 195)

    def _open_serial(self):
        """Prime serial port with stty and hold it open for the script lifetime."""
        try:
            subprocess.run(
                ['stty', '-F', SERIAL_DEVICE,
                 str(SERIAL_BAUD), 'cs8', '-cstopb', '-parenb', 'raw', '-hupcl'],
                check=False, capture_output=True)
        except Exception:
            pass
        if not _SERIAL_OK:
            self.get_logger().warn('pyserial not available — serial disabled')
            return
        try:
            self._serial = _serial_mod.Serial(
                SERIAL_DEVICE, SERIAL_BAUD,
                bytesize=8, parity='N', stopbits=1, timeout=0.1,
                dsrdtr=False, rtscts=False)
            self._serial.dtr = False
            self.get_logger().info(f'Serial open: {SERIAL_DEVICE}')
        except Exception as e:
            self.get_logger().warn(f'Serial open failed: {e}')

    def _send_pickup(self):
        self.get_logger().info('SERIAL: PICKUP (0x01)')
        if self._serial:
            try:
                self._serial.write(CMD_PICKUP)
            except Exception as e:
                self.get_logger().error(f'SERIAL: pickup failed: {e}')
        else:
            self.get_logger().warn('SERIAL: no port — pickup skipped')

    def _send_drop(self):
        self.get_logger().info('SERIAL: DROP (0x02)')
        if self._serial:
            try:
                self._serial.write(CMD_DROP)
            except Exception as e:
                self.get_logger().error(f'SERIAL: drop failed: {e}')
        else:
            self.get_logger().warn('SERIAL: no port — drop skipped')

    def turn_cw(self, degrees: float = DROP_DEG):
        """Turn clockwise by degrees using IMU yaw tracking."""
        target  = math.radians(degrees)
        ang_vel = -TURN_SPEED   # CW = negative angular in ROS2
        self.get_logger().info(f'TURN: {degrees:.0f} deg CW')

        start_yaw = None
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            start_yaw = self._get_yaw()
            if start_yaw is not None:
                break

        if start_yaw is None:
            self.get_logger().warn('TURN: no IMU — timed fallback')
            cmd = Twist()
            cmd.angular.z = ang_vel
            t0 = time.time()
            while time.time() - t0 < target / TURN_SPEED:
                self.cmd_pub.publish(cmd)
                rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(DT)
            self._stop()
            return

        cmd = Twist()
        cmd.angular.z = ang_vel
        rotated  = 0.0
        last_yaw = start_yaw
        while rotated < target:
            self.cmd_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(DT)
            yaw = self._get_yaw()
            if yaw is not None:
                rotated += abs(self._normalize_angle(yaw - last_yaw))
                last_yaw = yaw
        self._stop()
        time.sleep(0.2)
        self.get_logger().info(f'TURN: done ({math.degrees(rotated):.1f} deg)')

    def _get_best_cargo(self):
        if time.time() - self.last_det_time > 1.0:
            return None
        best = None
        for d in self.last_detections:
            if d.get('label') != 'cargo':
                continue
            if d.get('confidence', 0) < CV_MIN_CONFIDENCE:
                continue
            if best is None or d.get('area', 0) > best.get('area', 0):
                best = d
        return best

    def _detection_offset_x(self, det):
        return det['center'][0] - CV_CENTER_X

    def _detection_bbox_bottom(self, det):
        bbox = det.get('bbox', [0, 0, 0, 0])
        return bbox[1] + bbox[3]

    # ─── Scan ───────────────────────────────────────────────────
    def scan_for_cargo(self):
        """Slow 360 scan for cargo."""
        self.get_logger().info('SCAN: Looking for cargo...')
        t0 = time.time()
        cmd = Twist()
        total_rotated = 0.0
        last_yaw = None

        while rclpy.ok() and time.time() - t0 < CV_SCAN_TIMEOUT:
            self._spin(0.1)
            det = self._get_best_cargo()
            if det is not None:
                self._stop()
                self.get_logger().info(
                    f'SCAN: Found! conf={det["confidence"]:.2f} '
                    f'area={det["area"]:.0f} '
                    f'aspect={det.get("aspect_ratio", 0):.2f}')
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
                    break
            time.sleep(DT)

        self._stop()
        self.get_logger().warn('SCAN: Not found')
        return None

    # ─── Center ─────────────────────────────────────────────────
    def center_on_cargo(self, timeout=10.0):
        """Rotate to center cargo in frame. Returns last det or None."""
        self.get_logger().info('CENTER: Centering on cargo...')
        t0 = time.time()
        cmd = Twist()
        lost = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.1)
            det = self._get_best_cargo()
            if det is None:
                lost += 1
                if lost < 15:  # wait ~0.5s before drifting
                    time.sleep(DT)
                    continue
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

            spd = min(CV_ROTATE, max(0.04, abs(offset) / CAM_WIDTH * 0.5))
            cmd.angular.z = -spd if offset > 0 else spd
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)

        self._stop()
        self.get_logger().warn('CENTER: Timeout')
        return None

    # ─── Fine align ─────────────────────────────────────────────
    def fine_align(self, timeout=5.0):
        """Tight centering — 30px tolerance, 5 consecutive good frames."""
        self.get_logger().info('ALIGN: Fine alignment...')
        t0 = time.time()
        cmd = Twist()
        good = 0
        TIGHT = 30

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.1)
            det = self._get_best_cargo()
            if det is None:
                time.sleep(DT)
                continue

            offset = self._detection_offset_x(det)
            if abs(offset) < TIGHT:
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

    # ─── Approach for pickup ────────────────────────────────────
    def approach_for_pickup(self, timeout=25.0):
        """Drive toward cargo keeping centered. Stop when bbox bottom
        reaches PICKUP_BBOX_BOTTOM_Y, then nudge forward to engage.
        If cargo disappears from frame for ~1.5s, assume picked up."""
        self.get_logger().info(
            f'PICKUP: Approaching — stop at bbox_bot >= {PICKUP_BBOX_BOTTOM_Y}')
        t0 = time.time()
        cmd = Twist()
        lost_count = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin(0.05)

            front_min = self._get_front_min()
            if front_min < LIDAR_FRONT_STOP:
                self._stop()
                self.get_logger().info(
                    f'PICKUP: Lidar stop ({front_min:.2f}m)')
                return True

            det = self._get_best_cargo()
            if det is None:
                lost_count += 1
                if lost_count > 45:  # ~1.5s
                    self._stop()
                    self.get_logger().info('PICKUP: Cargo left frame — done!')
                    return True
                # Keep creeping forward
                cmd.linear.x = CV_LINEAR
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue
            lost_count = 0

            bbox_bot = self._detection_bbox_bottom(det)
            offset = self._detection_offset_x(det)

            self.get_logger().info(
                f'    bbox_bot={bbox_bot} offset={offset:.0f} '
                f'area={det["area"]:.0f} front={front_min:.2f}')

            if bbox_bot >= PICKUP_BBOX_BOTTOM_Y:
                self.get_logger().info(
                    f'PICKUP: bbox_bot={bbox_bot} — prongs at washer!')
                # Drive forward to fully push over cargo
                nudge = Twist()
                nudge.linear.x = CV_LINEAR * 0.5
                nt0 = time.time()
                while time.time() - nt0 < 7.0:
                    rclpy.spin_once(self, timeout_sec=0.05)
                    self.cmd_pub.publish(nudge)
                    time.sleep(DT)
                self._stop()
                # Pause over cargo before sending pickup
                self.get_logger().info('PICKUP: Pausing over cargo...')
                nt0 = time.time()
                while time.time() - nt0 < 1.5:
                    rclpy.spin_once(self, timeout_sec=0.1)
                # Send serial pickup command, wait 1s for actuator to drop
                self._send_pickup()
                self.get_logger().info('PICKUP: Waiting for actuator to drop...')
                nt0 = time.time()
                while time.time() - nt0 < 1.0:
                    rclpy.spin_once(self, timeout_sec=0.1)
                # Sweep: CCW 30deg → CW 60deg (past centre) → CCW 30deg back
                self.get_logger().info('PICKUP: Sweep rotation to engage...')
                SWEEP_SPD = 0.55   # rad/s
                sweep = Twist()
                for ang_vel, duration in [(SWEEP_SPD, 1.5),   # CCW 30deg
                                          (-SWEEP_SPD, 3.0),  # CW 60deg
                                          (SWEEP_SPD, 1.5)]:  # CCW 30deg back
                    sweep.angular.z = ang_vel
                    sweep.linear.x = 0.0
                    t0 = time.time()
                    while time.time() - t0 < duration:
                        self.cmd_pub.publish(sweep)
                        rclpy.spin_once(self, timeout_sec=0.05)
                        time.sleep(DT)
                self._stop()
                self.get_logger().info('PICKUP: Engaged!')
                return True

            # Steering correction
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
        self.get_logger().warn('PICKUP: Timeout')
        return False

    # ─── Reverse ────────────────────────────────────────────────
    def reverse_out(self, duration=3.0):
        self.get_logger().info(f'REVERSE: {duration:.1f}s')
        cmd = Twist()
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)
            rear = self._get_rear_min()
            if rear < LIDAR_FRONT_STOP:
                break
            cmd.linear.x = -REVERSE_SPEED
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)
        self._stop()

    def wait_for_cv(self, timeout=10.0):
        self.get_logger().info('Waiting for CV...')
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.last_det_time > 0:
                self.get_logger().info('CV active!')
                return True
        self.get_logger().warn('CV not detected — continuing')
        return False


def main():
    rclpy.init()
    node = CargoPickupTest()

    node.get_logger().info('===== CARGO PICKUP TEST =====')
    node.get_logger().info('Start robot facing the drop wall')

    node.wait_for_cv()
    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.1)

    # Step 1: Drop carried cargo
    node.get_logger().info('--- Step 1: Drop cargo ---')
    node._send_drop()
    t0 = time.time()
    while time.time() - t0 < DROP_PAUSE:
        rclpy.spin_once(node, timeout_sec=0.1)

    # Step 2: Rotate CW to face pickup cargo
    node.get_logger().info(f'--- Step 2: Turn {DROP_DEG:.0f} deg CW ---')
    node.turn_cw(DROP_DEG)

    # Step 3: Find cargo
    node.get_logger().info('--- Step 3: Scan ---')
    det = node.scan_for_cargo()
    if det is None:
        node.get_logger().error('No cargo found — aborting')
        node._stop()
        node.destroy_node()
        rclpy.shutdown()
        return

    # Step 4: Center
    node.get_logger().info('--- Step 4: Center ---')
    node.center_on_cargo()

    # Step 5: Fine align
    node.get_logger().info('--- Step 5: Fine align ---')
    node.fine_align()

    # Step 6: Approach and pick up
    node.get_logger().info('--- Step 6: Approach & pickup ---')
    result = node.approach_for_pickup()
    if result:
        node.get_logger().info('>>> CARGO PICKED UP <<<')
    else:
        node.get_logger().warn('Pickup may have failed')

    # Step 7: Reverse out
    node.get_logger().info('--- Step 7: Reverse ---')
    node.reverse_out(3.0)

    node.get_logger().info('===== PICKUP TEST COMPLETE =====')
    node._stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
