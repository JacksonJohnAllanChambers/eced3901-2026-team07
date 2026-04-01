#!/usr/bin/env python3
"""
full_lifeload.py — Right Open Water: basket pickup + cargo drop + lifeboat capture
===================================================================================
Combines full_load_right.py with lifeboat detection from open_water_right_lifeboat.py.

Sequence:
  Phase 1:  Nav2 drives from home to port approach zone
  Phase 2:  Blind drive forward into port entry
  Phase 3:  Turn CCW ~89.7° to face into port (basket orientation)
  Phase 4:  BASKET GRAB (0x04) — pick up cargo
  Phase 5:  Turn CW ~109.7° to face cargo drop zone
  Phase 6:  CARGO DROP (0x02) — release cargo
  Phase 7:  Reverse out of port
  Phase 8:  Turn CW ~180° to face home
  Phase 9:  Nav2 drives home
  Phase 10: Blind drive into home port
  Phase 11: Turn CW 180° — position back to face arena
  Phase 12: BASKET RELEASE (0x02) — release basket at home port
  Phase 13: Launch lifeboat detection camera node (dalibot_cv)
  Phase 14: Nav2 to approach zone while scanning for lifeboat
            → on detection: CV centre, CV approach, reverse + capture, resume Nav2
  Phase 15: Turn CCW 180° to face home
  Phase 16: Nav2 home

Usage:
  Terminal 1: ros2 launch dalmotor robot.launch.py
  Terminal 2: ros2 launch eced3901 arena_real_amcl.launch.py lane:=right_open
  Terminal 3: ros2 run eced3901 full_lifeload.py
  (Lifeboat camera node is launched automatically in Phase 13)
"""

import json
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
from std_msgs.msg import String
import tf2_ros

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

HOME_X   = 0.08
HOME_Y   = -1.08
HOME_YAW = 0.0

APPROACH_X   = 3.40
APPROACH_Y   = -0.98   # shifted 10 cm towards arena centre (was -1.08)
APPROACH_YAW = 0.0

NAV2_PREEMPT_DIST = 0.25   # cancel Nav2 when this close to approach goal

# ── Phase 2: blind drive into port entry ─────────────────────
# Original: 0.52 m @ 0.08 m/s = 6.5 s; +5 cm deeper → 0.57 m @ 0.05 m/s = 11.4 s
DRIVE_SPEED      = 0.05    # m/s (slowed from 0.08)
DRIVE_TIME       = 10.9    # seconds to reach port entry (+5 cm) from Nav2 handoff
DRIVE_LIDAR_STOP = 0.15    # emergency front stop

# ── Phase 3: CCW turn to basket grab orientation ─────────────
# Recording: yaw -6° → 93.7° = 99.7° CCW = 1.74 rad @ 0.25 rad/s = 7.0 s
# Adjusted: -10° → 89.7° CCW = 1.57 rad @ 0.25 rad/s = 6.3 s
TURN_SPEED_SLOW = 0.25     # rad/s (slowed from 0.35)
TURN_CCW_TIME   = 6.3      # seconds (reduced by 10°)

# ── Phase 4: basket grab ─────────────────────────────────────
BASKET_WAIT = 1.5          # seconds after sending 0x04

# ── Phase 5: CW turn back to cargo drop orientation ──────────
# Recording: 93.7° → -6° = 99.7° CW = 1.74 rad @ 0.25 rad/s = 7.0 s
# Adjusted: +10° → 109.7° CW = 1.91 rad @ 0.25 rad/s = 7.7 s
TURN_CW_PARTIAL_TIME = 7.7  # seconds (increased by 10°)

# ── Phase 6: cargo drop ───────────────────────────────────────
DROP_WAIT = 2.5            # seconds after sending 0x02

# ── Phase 7: reverse out of port ─────────────────────────────
# Original: 0.40 m @ 0.08 m/s = 5.0 s; +5 cm deeper → 0.45 m @ 0.05 m/s = 9.0 s
# Extra reverse added before turn: 9.0 → 12.0 s (~0.15 m more clearance)
REVERSE_SPEED      = 0.05   # m/s
REVERSE_TIME       = 5.0    # seconds
REVERSE_LIDAR_STOP = 0.12   # rear safety stop

# ── Phase 8: CCW turn to face home ───────────────────────────
# -6° → ~180° ≈ 174° CCW @ 0.30 rad/s = 10.8 s
TURN_SPEED_FAST = 0.30      # rad/s (slowed from 0.40)
TURN_180_TIME   = 10.8

# ── Phase 10: blind drive into home port ─────────────────────
HOME_DRIVE_TIME  = 4.5     # seconds to push into home port after Nav2 (+15 cm nudge)

# ── Phase 12: basket release at home port ────────────────────
BASKET_RELEASE_WAIT = 1.5  # seconds after sending release command

# ── Lifeboat detection (Phases 13–14) ────────────────────────
CAM_WIDTH              = 1280
CV_CENTER_TOL          = 60    # pixels — acceptable centre offset
CV_ROTATE              = 0.20  # rad/s for centering rotation
CV_LINEAR              = 0.10  # m/s for CV approach
CV_LIFEBOAT_AREA_CLOSE = 8000  # stop approaching when blob area >= this
LIFEBOAT_MIN_AREA      = 300   # ignore detections smaller than this
LIFEBOAT_MIN_CONF      = 0.35
CV_CENTER_TIMEOUT      = 10.0  # s
CV_APPROACH_TIMEOUT    = 25.0  # s
LIDAR_FRONT_STOP       = 0.12  # hard lidar stop during CV approach
CAPTURE_REVERSE_TIME   = 1.0   # seconds to back into lifeboat

# ── Serial ───────────────────────────────────────────────────
SERIAL_PORTS          = ['/dev/ttyUSB0', '/dev/ttyUSB4']
SERIAL_BAUD           = 9600
CMD_BASKET_GRAB       = b'\x04'
CMD_DROP              = b'\x02'
CMD_BASKET_RELEASE    = b'\x03'
CMD_LIFEBOAT_CAPTURE  = b'\x04'

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
        super().__init__('full_lifeload')
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.last_scan = None
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, scan_qos)

        self.last_detections = []
        self.create_subscription(String, '/cv/detections', self._det_cb, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._serial = None
        self._open_serial()

    # ── Callbacks ────────────────────────────────────────────
    def _scan_cb(self, msg):
        self.last_scan = msg

    def _det_cb(self, msg):
        try:
            self.last_detections = json.loads(msg.data)
        except json.JSONDecodeError:
            self.last_detections = []

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

    # ── CV helpers ───────────────────────────────────────────
    def get_best_lifeboat(self):
        """Return highest-confidence valid lifeboat detection or None."""
        best = None
        for d in self.last_detections:
            if d.get('label') != 'lifeboat':
                continue
            if d.get('confidence', 0) < LIFEBOAT_MIN_CONF:
                continue
            if d.get('area', 0) < LIFEBOAT_MIN_AREA:
                continue
            if best is None or d['confidence'] > best['confidence']:
                best = d
        return best

    def cv_center_lifeboat(self):
        """Rotate in place until lifeboat is centred in frame."""
        self.get_logger().info('LIFEBOAT: Centering...')
        t0 = time.time()
        cmd = Twist()
        while time.time() - t0 < CV_CENTER_TIMEOUT:
            rclpy.spin_once(self, timeout_sec=0.05)
            det = self.get_best_lifeboat()
            if det is None:
                cmd.angular.z = CV_ROTATE
                cmd.linear.x = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue
            offset = det['center'][0] - CAM_WIDTH / 2
            if abs(offset) <= CV_CENTER_TOL:
                self.stop()
                self.get_logger().info(f'  Centred! offset={offset:.0f}px')
                return True
            cmd.angular.z = -offset / (CAM_WIDTH / 2) * CV_ROTATE
            cmd.angular.z = max(-CV_ROTATE, min(CV_ROTATE, cmd.angular.z))
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)
        self.stop()
        self.get_logger().warn('LIFEBOAT: Centering timeout')
        return False

    def cv_approach_lifeboat(self):
        """Drive toward lifeboat keeping it centred until close."""
        self.get_logger().info('LIFEBOAT: Approaching...')
        t0 = time.time()
        cmd = Twist()
        lost_count = 0
        while time.time() - t0 < CV_APPROACH_TIMEOUT:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.get_front() < LIDAR_FRONT_STOP:
                self.stop()
                self.get_logger().info('  Lidar front stop')
                return True
            det = self.get_best_lifeboat()
            if det is None:
                lost_count += 1
                if lost_count > 45:
                    self.stop()
                    self.get_logger().warn('  Lost lifeboat during approach')
                    return False
                cmd.linear.x = CV_LINEAR * 0.3
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue
            lost_count = 0
            area = det.get('area', 0)
            if area >= CV_LIFEBOAT_AREA_CLOSE:
                self.stop()
                self.get_logger().info(f'  Close enough (area={area:.0f})')
                return True
            offset = det['center'][0] - CAM_WIDTH / 2
            steer = -offset / (CAM_WIDTH / 2) * CV_ROTATE
            steer = max(-CV_ROTATE, min(CV_ROTATE, steer))
            cmd.linear.x = CV_LINEAR
            cmd.angular.z = steer
            self.cmd_pub.publish(cmd)
            time.sleep(DT)
        self.stop()
        self.get_logger().warn('LIFEBOAT: Approach timeout')
        return False

    def do_lifeboat_capture(self):
        """Full lifeboat capture sequence from current position."""
        print('\n  ── Lifeboat detected! Starting capture ──')

        # Centre on lifeboat
        self.cv_center_lifeboat()

        # Drive up to it
        self.cv_approach_lifeboat()
        self.print_pose('LIFEBOAT PRE-CAPTURE')

        # Turn CW 180° so back faces lifeboat
        print('  Turning CW to back up...')
        self.timed_turn(-TURN_SPEED_FAST, TURN_180_TIME)

        # Reverse into lifeboat
        print('  Reversing into lifeboat...')
        self.blind_drive(-REVERSE_SPEED, CAPTURE_REVERSE_TIME)

        # Send capture command
        self.send_serial(CMD_LIFEBOAT_CAPTURE, 'LIFEBOAT CAPTURE (0x04)')
        print('  Capture command sent — waiting...')
        self.spin_ros(1.0)

        # Turn CCW 180° to face port again
        print('  Turning CCW back to face approach zone...')
        self.timed_turn(+TURN_SPEED_FAST, TURN_180_TIME)

        self.print_pose('LIFEBOAT POST-CAPTURE')
        print('  ── Lifeboat capture complete — resuming Nav2 ──\n')

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
    def blind_drive(self, speed, duration, lidar_field=None, lidar_stop=None):
        """Drive forward (speed>0) or backward (speed<0) for duration seconds."""
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
    print('   RIGHT OPEN WATER — FULL LOAD + LIFEBOAT')
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
    #  PHASE 2 — Drive forward into port entry
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 2: Drive into port entry ═══')
    m.blind_drive(DRIVE_SPEED, DRIVE_TIME,
                  lidar_field='front', lidar_stop=DRIVE_LIDAR_STOP)
    m.print_pose('PORT ENTRY')

    # ══════════════════════════════════════════════════════════
    #  PHASE 3 — Turn CCW ~89.7° to basket grab orientation
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 3: Turn CCW into port (basket orientation) ═══')
    m.timed_turn(+TURN_SPEED_SLOW, TURN_CCW_TIME)
    m.print_pose('BASKET POSITION')

    # ══════════════════════════════════════════════════════════
    #  PHASE 4 — Basket grab
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 4: Basket grab ═══')
    m.send_serial(CMD_BASKET_GRAB, 'BASKET GRAB (0x04)')
    print('  Waiting for basket to engage...')
    m.spin_ros(BASKET_WAIT)
    m.print_pose('AFTER GRAB')

    # ══════════════════════════════════════════════════════════
    #  PHASE 5 — Turn CW ~109.7° to cargo drop zone
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 5: Turn CW to cargo drop zone ═══')
    m.timed_turn(-TURN_SPEED_SLOW, TURN_CW_PARTIAL_TIME)
    m.print_pose('DROP POSITION')

    # ══════════════════════════════════════════════════════════
    #  PHASE 6 — Drop cargo
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 6: Drop cargo ═══')
    m.send_serial(CMD_DROP, 'CARGO DROP (0x02)')
    print('  Waiting for cargo to release...')
    t0 = time.time()
    while time.time() - t0 < DROP_WAIT:
        m.spin_ros(0.1)
    m.print_pose('AFTER DROP')

    # ══════════════════════════════════════════════════════════
    #  PHASE 7 — Reverse out of port
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 7: Reverse out of port ═══')
    m.blind_drive(-REVERSE_SPEED, REVERSE_TIME,
                  lidar_field='rear', lidar_stop=REVERSE_LIDAR_STOP)
    m.print_pose('REVERSED')

    # ══════════════════════════════════════════════════════════
    #  PHASE 8 — Turn CW ~180° to face home
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 8: Turn CCW ~180° to face home ═══')
    m.timed_turn(-TURN_SPEED_FAST, TURN_180_TIME)
    m.print_pose('TURNED')

    # ══════════════════════════════════════════════════════════
    #  PHASE 9 — Nav2 home
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 9: Nav2 → home ═══')
    nav.goToPose(create_pose(nav, HOME_X, HOME_Y, 180.0))

    while not nav.isTaskComplete():
        rclpy.spin_once(m, timeout_sec=0.05)
        fb = nav.getFeedback()
        if fb:
            dist = fb.distance_remaining
            print(f'  dist remaining: {dist:.2f}m   ', end='\r')

    print(f'\n  Nav2 result: {nav.getResult()}')
    m.print_pose('HOME')

    # ══════════════════════════════════════════════════════════
    #  PHASE 10 — Blind drive into home port
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 10: Drive into home port ═══')
    m.blind_drive(DRIVE_SPEED, HOME_DRIVE_TIME,
                  lidar_field='front', lidar_stop=DRIVE_LIDAR_STOP)
    m.print_pose('HOME PORT')

    # ══════════════════════════════════════════════════════════
    #  PHASE 11 — Turn CW 180° to face arena
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 11: Turn CW 180° to face arena ═══')
    m.timed_turn(-TURN_SPEED_FAST, TURN_180_TIME)
    m.print_pose('TURNED TO ARENA')

    # ══════════════════════════════════════════════════════════
    #  PHASE 12 — Release basket at home port
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 12: Release basket ═══')
    m.send_serial(CMD_BASKET_RELEASE, 'BASKET RELEASE (0x03)')
    print('  Waiting for basket to release...')
    m.spin_ros(BASKET_RELEASE_WAIT)
    m.print_pose('AFTER BASKET RELEASE')

    # ══════════════════════════════════════════════════════════
    #  PHASE 13 — Launch lifeboat detection camera node
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 13: Launching lifeboat detection camera node ═══')
    cam_proc = subprocess.Popen(
        ['ros2', 'launch', 'dalibot_cv', 'lifeboat_only_launch.py'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    print('  Camera node launched — waiting 3 s for it to come up...')
    time.sleep(3.0)
    print('  Camera node ready')

    # ══════════════════════════════════════════════════════════
    #  PHASE 14 — Nav2 to approach zone, scanning for lifeboat
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 14: Nav2 → approach zone (scanning for lifeboat) ═══')

    approach_goal = create_pose(nav, APPROACH_X, APPROACH_Y, APPROACH_YAW)
    lifeboat_captured = False

    nav.goToPose(approach_goal)
    while True:
        rclpy.spin_once(m, timeout_sec=0.05)

        if nav.isTaskComplete():
            print(f'  Nav2 result: {nav.getResult()}')
            break

        fb = nav.getFeedback()
        if fb:
            dist = fb.distance_remaining
            print(f'  dist remaining: {dist:.2f}m  '
                  + ('[LIFEBOAT DONE]' if lifeboat_captured else '[scanning...]'),
                  end='\r')
            if dist < NAV2_PREEMPT_DIST:
                print(f'\n  Close enough ({dist:.2f}m) — cancelling Nav2')
                nav.cancelTask()
                break

        # Check for lifeboat only if not yet captured
        if not lifeboat_captured:
            det = m.get_best_lifeboat()
            if det is not None:
                print(f'\n  Lifeboat! conf={det["confidence"]:.2f} area={det["area"]:.0f}')
                nav.cancelTask()
                m.do_lifeboat_capture()
                lifeboat_captured = True
                # Re-issue Nav2 goal to continue to approach zone
                approach_goal = create_pose(nav, APPROACH_X, APPROACH_Y, APPROACH_YAW)
                nav.goToPose(approach_goal)

    time.sleep(0.5)
    m.print_pose('APPROACH ZONE')

    # ══════════════════════════════════════════════════════════
    #  PHASE 15 — Turn CCW 180° to face home
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 15: Turn CCW 180° to face home ═══')
    m.timed_turn(+TURN_SPEED_FAST, TURN_180_TIME)
    m.print_pose('TURNED TO HOME')

    # ══════════════════════════════════════════════════════════
    #  PHASE 16 — Nav2 home
    # ══════════════════════════════════════════════════════════
    print('\n═══ PHASE 16: Nav2 → home ═══')
    nav.goToPose(create_pose(nav, HOME_X, HOME_Y, 180.0))

    while not nav.isTaskComplete():
        rclpy.spin_once(m, timeout_sec=0.05)
        fb = nav.getFeedback()
        if fb:
            dist = fb.distance_remaining
            print(f'  dist remaining: {dist:.2f}m   ', end='\r')

    print(f'\n  Nav2 result: {nav.getResult()}')
    m.print_pose('HOME')

    print()
    print('══════════════════════════════════════════════════')
    if lifeboat_captured:
        print('   MISSION COMPLETE — lifeboat captured!')
    else:
        print('   MISSION COMPLETE — lifeboat not seen')
    print('══════════════════════════════════════════════════')
    print()

    cam_proc.terminate()
    m.stop()
    if m._serial:
        m._serial.close()
    m.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
