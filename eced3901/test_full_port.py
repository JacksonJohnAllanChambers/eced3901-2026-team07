#!/usr/bin/env python3
"""
test_full_port.py — Full port interaction test
===============================================
Start robot ~1 foot (30 cm) outside the target port, facing the port.

Sequence:
  1. Drive forward ~30 cm into the port
  2. Turn 90 deg left or right (--side arg)
  3. Drop cargo via serial (0x02)
  4. Turn back 90 deg to original heading
  5. Scan, center, fine-align on target cargo
  6. CV approach to cargo
  7. Drive forward over cargo, pause, pickup via serial (0x01)
  8. Jiggle sweep to engage
  9. Reverse out

Usage:
  ros2 launch dalmotor robot.launch.py
  ros2 launch dalibot_cv dalibot_cv_launch.py
  ros2 run eced3901 test_full_port.py --ros-args -p side:=left
  ros2 run eced3901 test_full_port.py --ros-args -p side:=right
"""

import argparse
import json
import math
import subprocess
import sys
import time

try:
    import serial as _serial_mod
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

SERIAL_DEVICE = '/dev/ttyUSB0'
SERIAL_BAUD   = 9600
CMD_DROP      = b'\x02'
CMD_PICKUP    = b'\x01'

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String

# ── Speeds ──────────────────────────────────────────────────────────
CV_LINEAR     = 0.06
CV_ROTATE     = 0.15
REVERSE_SPEED = 0.08
TURN_SPEED    = 0.55   # rad/s for 90 deg turns and sweep

# ── Entry drive ─────────────────────────────────────────────────────
ENTRY_SPEED    = 0.10   # m/s
ENTRY_DURATION = 2.0    # s  (~20 cm at 0.10 m/s)

# ── Turn angles ─────────────────────────────────────────────────────
TURN_DROP_DEG    = 65.0   # deg for drop turn
TURN_BACK_DEG    = 65.0   # deg to turn back

# ── Drop timing ─────────────────────────────────────────────────────
APPROACH_BEFORE_DROP_SECS = 1.5   # s to drive toward cargo before dropping
DROP_PAUSE    = 3.0    # s after drop command

# ── Lidar ───────────────────────────────────────────────────────────
LIDAR_FRONT_STOP = 0.12
LIDAR_SLOW_DIST  = 0.40

# ── CV ──────────────────────────────────────────────────────────────
CV_CENTER_X       = 640
CV_CENTER_TOL     = 60
CV_MIN_CONFIDENCE = 0.35
CV_SCAN_TIMEOUT   = 20.0
CAM_WIDTH         = 1280

# ── Pickup threshold ────────────────────────────────────────────────
PICKUP_BBOX_BOTTOM_Y = 580

DT = 1.0 / 30.0


class FullPortTest(Node):
    def __init__(self, side: str):
        super().__init__('full_port_test')
        self.side = side   # 'left' or 'right'

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.last_odom = None
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.last_imu = None
        self.create_subscription(Imu, '/bno055/imu', self._imu_cb, 10)
        self.last_scan = None
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, scan_qos)
        self.last_detections = []
        self.last_det_time = 0.0
        self.create_subscription(String, '/cv/detections', self._det_cb, 10)

        self._serial = None
        self._open_serial()

    def _odom_cb(self, msg): self.last_odom = msg
    def _imu_cb(self, msg):  self.last_imu  = msg
    def _scan_cb(self, msg): self.last_scan  = msg

    def _det_cb(self, msg):
        try:
            self.last_detections = json.loads(msg.data)
            self.last_det_time = time.time()
        except json.JSONDecodeError:
            self.last_detections = []

    # ── Helpers ─────────────────────────────────────────────────────

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
        while a >  math.pi: a -= 2 * math.pi
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
            r = scan.ranges[deg % n]
            if scan.range_min < r < scan.range_max:
                r_min = min(r_min, r)
        return r_min

    def _get_front_min(self):
        return min(self._lidar_sector_min(0, 15),
                   self._lidar_sector_min(345, 359))

    def _get_rear_min(self):
        return self._lidar_sector_min(165, 195)

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

    # ── Serial ──────────────────────────────────────────────────────

    def _open_serial(self):
        """Prime port with stty then open and hold for duration of script."""
        try:
            subprocess.run(
                ['stty', '-F', SERIAL_DEVICE,
                 '9600', 'cs8', '-cstopb', '-parenb', 'raw', '-hupcl'],
                check=True)
            self.get_logger().info('stty: serial port primed')
        except Exception as e:
            self.get_logger().warn(f'stty failed: {e}')

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
            self._serial = None

    def _send_serial(self, data: bytes, label: str):
        self.get_logger().info(f'SERIAL: {label}')
        if self._serial:
            try:
                self._serial.write(data)
            except Exception as e:
                self.get_logger().error(f'SERIAL write failed: {e}')
        else:
            self.get_logger().warn(f'No serial port — {label} skipped')

    # ── Motion primitives ────────────────────────────────────────────

    def drive_forward(self, duration: float, speed: float = ENTRY_SPEED):
        """Drive forward for a fixed duration."""
        self.get_logger().info(f'DRIVE: forward {duration:.1f}s at {speed:.2f} m/s')
        cmd = Twist()
        cmd.linear.x = speed
        t0 = time.time()
        while time.time() - t0 < duration:
            self.cmd_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(DT)
        self._stop()
        time.sleep(0.2)

    def turn_90(self, direction: str, degrees: float = 90.0):
        """Turn by `degrees` left (CCW) or right (CW) using IMU yaw tracking."""
        target = math.radians(degrees)
        ang_vel = TURN_SPEED if direction == 'left' else -TURN_SPEED
        self.get_logger().info(f'TURN: {degrees:.0f} deg {direction}')

        # Wait for a valid yaw reading
        start_yaw = None
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            start_yaw = self._get_yaw()
            if start_yaw is not None:
                break

        if start_yaw is None:
            self.get_logger().warn('TURN: No IMU/odom — falling back to timed turn')
            cmd = Twist()
            cmd.angular.z = ang_vel
            t0 = time.time()
            timed = target / abs(ang_vel)
            while time.time() - t0 < timed:
                self.cmd_pub.publish(cmd)
                rclpy.spin_once(self, timeout_sec=0.05)
                time.sleep(DT)
            self._stop()
            return

        cmd = Twist()
        cmd.angular.z = ang_vel
        rotated = 0.0
        last_yaw = start_yaw

        while rotated < target:
            self.cmd_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(DT)
            yaw = self._get_yaw()
            if yaw is not None:
                delta = abs(self._normalize_angle(yaw - last_yaw))
                rotated += delta
                last_yaw = yaw

        self._stop()
        time.sleep(0.2)
        self.get_logger().info(f'TURN: done ({math.degrees(rotated):.1f} deg)')

    def reverse_out(self, duration=3.0):
        self.get_logger().info(f'REVERSE: {duration:.1f}s')
        cmd = Twist()
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._get_rear_min() < LIDAR_FRONT_STOP:
                break
            cmd.linear.x = -REVERSE_SPEED
            self.cmd_pub.publish(cmd)
            time.sleep(DT)
        self._stop()

    # ── CV sequence (from test_port_interaction.py) ──────────────────

    def scan_for_cargo(self):
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
                    f'SCAN: Found! conf={det["confidence"]:.2f} area={det["area"]:.0f}')
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

    def center_on_cargo(self, timeout=10.0):
        self.get_logger().info('CENTER: Centering on cargo...')
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
                cmd.angular.z = CV_ROTATE * 0.3
                cmd.linear.x = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue
            lost = 0
            offset = self._detection_offset_x(det)
            if abs(offset) < CV_CENTER_TOL:
                self._stop()
                self.get_logger().info(f'CENTER: Done (offset={offset:.0f}px)')
                return det
            spd = min(CV_ROTATE, max(0.04, abs(offset) / CAM_WIDTH * 0.5))
            cmd.angular.z = -spd if offset > 0 else spd
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)

        self._stop()
        self.get_logger().warn('CENTER: Timeout')
        return None

    def fine_align(self, timeout=5.0):
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
                    self.get_logger().info(f'ALIGN: Locked (offset={offset:.0f}px)')
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

    def approach_for_pickup(self, timeout=25.0):
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
                self.get_logger().info(f'PICKUP: Lidar stop ({front_min:.2f}m)')
                return True

            det = self._get_best_cargo()
            if det is None:
                lost_count += 1
                if lost_count > 45:
                    self._stop()
                    self.get_logger().info('PICKUP: Cargo left frame — done!')
                    return True
                cmd.linear.x = CV_LINEAR
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue
            lost_count = 0

            bbox_bot = self._detection_bbox_bottom(det)
            offset   = self._detection_offset_x(det)
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
                while time.time() - nt0 < 5.0:
                    rclpy.spin_once(self, timeout_sec=0.05)
                    self.cmd_pub.publish(nudge)
                    time.sleep(DT)
                self._stop()
                # Pause over cargo
                self.get_logger().info('PICKUP: Pausing over cargo...')
                nt0 = time.time()
                while time.time() - nt0 < 1.5:
                    rclpy.spin_once(self, timeout_sec=0.1)
                # Send pickup command, wait for actuator
                self._send_serial(CMD_PICKUP, 'PICKUP (0x01)')
                self.get_logger().info('PICKUP: Waiting for actuator to drop...')
                nt0 = time.time()
                while time.time() - nt0 < 1.0:
                    rclpy.spin_once(self, timeout_sec=0.1)
                # Sweep: CCW 30deg -> CW 60deg -> CCW 30deg back
                self.get_logger().info('PICKUP: Sweep to engage...')
                SWEEP_SPD = 0.55
                sweep = Twist()
                for ang_vel, duration in [(SWEEP_SPD, 1.5),
                                          (-SWEEP_SPD, 3.0),
                                          (SWEEP_SPD, 1.5)]:
                    sweep.angular.z = ang_vel
                    sweep.linear.x = 0.0
                    nt0 = time.time()
                    while time.time() - nt0 < duration:
                        self.cmd_pub.publish(sweep)
                        rclpy.spin_once(self, timeout_sec=0.05)
                        time.sleep(DT)
                self._stop()
                self.get_logger().info('PICKUP: Engaged!')
                return True

            steer = 0.0
            if abs(offset) > CV_CENTER_TOL * 0.5:
                steer = -offset / CAM_WIDTH * 0.8
                steer = max(-CV_ROTATE, min(CV_ROTATE, steer))
            speed = CV_LINEAR
            if front_min < LIDAR_SLOW_DIST:
                speed *= max(0.3, front_min / LIDAR_SLOW_DIST)
            cmd.linear.x  = speed
            cmd.angular.z = steer
            self.cmd_pub.publish(cmd)
            time.sleep(DT)

        self._stop()
        self.get_logger().warn('PICKUP: Timeout')
        return False

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

    # Read side parameter
    param_node = rclpy.create_node('param_reader')
    param_node.declare_parameter('side', 'left')
    side = param_node.get_parameter('side').value.lower()
    param_node.destroy_node()

    if side not in ('left', 'right'):
        print(f'Invalid side "{side}" — use left or right')
        rclpy.shutdown()
        return

    opposite = 'right' if side == 'left' else 'left'

    node = FullPortTest(side=side)
    node.get_logger().info(f'===== FULL PORT TEST  side={side.upper()} =====')
    node.get_logger().info('Robot should be ~30 cm outside the target port, facing it')

    node.wait_for_cv()
    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.1)

    # ── Step 1: Drive forward into port ─────────────────────────────
    node.get_logger().info('--- Step 1: Enter port ---')
    node.drive_forward(ENTRY_DURATION, ENTRY_SPEED)

    # ── Step 2: Turn toward drop zone ───────────────────────────────
    node.get_logger().info(f'--- Step 2: Turn {TURN_DROP_DEG:.0f} deg {side} ---')
    node.turn_90(side, degrees=TURN_DROP_DEG)

    # ── Step 2b: Lock onto cargo and approach before dropping ────────
    node.get_logger().info('--- Step 2b: Find cargo and approach ---')
    det = node.scan_for_cargo()
    if det is not None:
        node.center_on_cargo()
        node.drive_forward(APPROACH_BEFORE_DROP_SECS, CV_LINEAR)
    else:
        node.get_logger().warn('Step 2b: No cargo found — dropping from current position')

    # ── Step 3: Drop cargo ──────────────────────────────────────────
    node.get_logger().info('--- Step 3: Drop cargo ---')
    node._send_serial(CMD_DROP, 'DROP (0x02)')
    node.get_logger().info(f'Waiting {DROP_PAUSE:.0f}s for cargo to drop...')
    t0 = time.time()
    while time.time() - t0 < DROP_PAUSE:
        rclpy.spin_once(node, timeout_sec=0.1)

    # ── Step 4: Turn back to original heading ───────────────────────
    node.get_logger().info(f'--- Step 4: Turn {TURN_BACK_DEG:.0f} deg {opposite} (back) ---')
    node.turn_90(opposite, degrees=TURN_BACK_DEG)

    # ── Step 5: Scan for target cargo ───────────────────────────────
    node.get_logger().info('--- Step 5: Scan ---')
    det = node.scan_for_cargo()
    if det is None:
        node.get_logger().error('No cargo found — aborting')
        node._stop()
        node.destroy_node()
        rclpy.shutdown()
        return

    # ── Step 6: Center ──────────────────────────────────────────────
    node.get_logger().info('--- Step 6: Center ---')
    node.center_on_cargo()

    # ── Step 7: Fine align ──────────────────────────────────────────
    node.get_logger().info('--- Step 7: Fine align ---')
    node.fine_align()

    # ── Step 8: Approach and pick up ────────────────────────────────
    node.get_logger().info('--- Step 8: Approach & pickup ---')
    result = node.approach_for_pickup()
    if result:
        node.get_logger().info('>>> CARGO PICKED UP <<<')
    else:
        node.get_logger().warn('Pickup may have failed')

    # ── Step 9: Reverse out ─────────────────────────────────────────
    node.get_logger().info('--- Step 9: Reverse ---')
    node.reverse_out(3.0)

    node.get_logger().info('===== FULL PORT TEST COMPLETE =====')
    node._stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
