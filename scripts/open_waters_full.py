#!/usr/bin/env python3
"""
open_waters_full.py — Full open waters run built on simple_open_water.py fundamentals.

Adds on top of the working nav:
  * Serial cargo DROP (0x02) and PICKUP (0x01) commands
  * Back-up + side maneuver after drop
  * CV-guided approach to target cargo

Usage:
  ros2 launch dalmotor robot.launch.py
  ros2 launch eced3901 arena_amcl.launch.py lane:=right_open use_rviz:=False
  ros2 launch dalibot_cv dalibot_cv_launch.py
  ros2 run eced3901 open_waters_full.py --ros-args -p lane:=right_open
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String

try:
    import serial as _serial_mod
    _SERIAL_OK = True
except ImportError:
    _SERIAL_OK = False

# ── Serial ──────────────────────────────────────────────────────────
SERIAL_DEVICE  = '/dev/ttyUSB0'
SERIAL_BAUD    = 9600
CMD_DROP       = b'\x02'
CMD_PICKUP     = b'\x01'

# ── Timing ──────────────────────────────────────────────────────────
DROP_PAUSE     = 3.0    # s — wait after drop command
PICKUP_PAUSE   = 5.0    # s — wait after pickup command
BACKUP_TIME    = 2.5    # s — reverse after drop
BACKUP_SPEED   = -0.10  # m/s

# ── Maneuver (side step to approach position) ────────────────────────
# Robot arrives at port facing +y (north/90 deg).
# After backing up, step +x and face west (180 deg) to approach cargo.
MANEUVER_X_OFFSET =  0.35   # m east of port
MANEUVER_Y_OFFSET = -0.20   # m south of port
APPROACH_YAW_DEG  =  180.0  # face west (-x) toward cargo

# ── CV approach ─────────────────────────────────────────────────────
CV_AREA_TARGET  = 35000   # pixel area = close enough to pick up
CV_APPROACH_SPD =  0.08   # m/s linear while approaching
CV_TURN_GAIN    =  1.5    # angular vel (rad/s) per radian of bearing error
CV_MIN_CONF     =  0.30
CV_TIMEOUT      = 15.0    # s


# ════════════════════════════════════════════════════════════════════
# Helper node — serial + CV subscriptions + cmd_vel
# ════════════════════════════════════════════════════════════════════
class CargoHelper(Node):
    def __init__(self):
        super().__init__('cargo_helper')
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.last_detections = []
        self.create_subscription(String, '/cv/detections', self._det_cb, 10)
        self._serial = None
        # Serial is NOT opened here — open lazily at first send_cargo_cmd call
        # to avoid DTR toggle triggering the cargo mechanism at startup.

    def _det_cb(self, msg):
        try:
            self.last_detections = json.loads(msg.data)
        except Exception:
            self.last_detections = []

    def _open_serial(self):
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

    def send_cargo_cmd(self, data: bytes, label: str):
        if self._serial is None:
            self._open_serial()   # lazy init — only open serial when first needed
        self.get_logger().info(f'CARGO CMD: {label}')
        if self._serial:
            try:
                self._serial.write(data)
            except Exception as e:
                self.get_logger().error(f'Serial write error: {e}')
        else:
            self.get_logger().warn(f'No serial port — {label} skipped')

    def stop(self):
        self.cmd_pub.publish(Twist())

    def drive(self, linear: float, angular: float = 0.0):
        t = Twist()
        t.linear.x  = linear
        t.angular.z = angular
        self.cmd_pub.publish(t)

    def spin_pause(self, seconds: float):
        """Sleep while keeping ROS callbacks alive."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

    def best_cargo_detection(self):
        """Return highest-confidence cargo detection, or None."""
        rclpy.spin_once(self, timeout_sec=0.05)
        best = None
        for d in self.last_detections:
            if d.get('label') == 'cargo' and d.get('confidence', 0) >= CV_MIN_CONF:
                if best is None or d.get('area', 0) > best.get('area', 0):
                    best = d
        return best

    def cv_approach_cargo(self):
        """Drive toward cargo using bearing + area feedback. Returns True on success."""
        print('  CV approach: searching for cargo...')
        deadline  = time.time() + CV_TIMEOUT
        seen_once = False

        while time.time() < deadline:
            det = self.best_cargo_detection()

            if det is None:
                if seen_once:
                    self.stop()
                    time.sleep(0.3)
                else:
                    self.drive(0.05)   # creep forward until we see it
                continue

            seen_once = True
            area    = det.get('area', 0)
            bearing = det.get('bearing_deg', 0.0)   # positive = right of centre
            print(f'  area={area:.0f}  bearing={bearing:+.1f}deg', end='\r')

            if area >= CV_AREA_TARGET:
                self.stop()
                print(f'\n  CV approach done — area={area:.0f}')
                return True

            # P-control: positive bearing = cargo is right = turn right (neg angular)
            angular = -math.radians(bearing) * CV_TURN_GAIN
            angular = max(-0.4, min(0.4, angular))
            self.drive(CV_APPROACH_SPD, angular)

        self.stop()
        print('\n  CV approach timed out — blind nudge')
        self.drive(0.10)
        time.sleep(2.0)
        self.stop()
        return False


# ════════════════════════════════════════════════════════════════════
# Nav helpers
# ════════════════════════════════════════════════════════════════════
def make_pose(nav, x, y, yaw_deg):
    p = PoseStamped()
    p.header.frame_id = 'map'
    p.header.stamp    = nav.get_clock().now().to_msg()
    yaw = math.radians(yaw_deg)
    p.pose.position.x    = x
    p.pose.position.y    = y
    p.pose.orientation.z = math.sin(yaw / 2.0)
    p.pose.orientation.w = math.cos(yaw / 2.0)
    return p


def go_to(nav, helper, x, y, yaw_deg, label=''):
    print(f'  -> {label}  ({x:.3f}, {y:.3f})  yaw={yaw_deg:.0f}deg')
    nav.goToPose(make_pose(nav, x, y, yaw_deg))
    while not nav.isTaskComplete():
        rclpy.spin_once(helper, timeout_sec=0.1)


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
def main():
    rclpy.init()

    # Lane parameter — same as simple_open_water.py
    pnode = rclpy.create_node('lane_selector')
    pnode.declare_parameter('lane', 'left_open')
    lane = pnode.get_parameter('lane').value
    pnode.destroy_node()

    coords = {
        'left_open':  {'x': 1.523, 'y': 0.305, 'goal_y': 3.67, 'home_yaw': -90.0},
        'right_open': {'x': 2.742, 'y': 0.305, 'goal_y': 3.67, 'home_yaw':  90.0},
    }
    data   = coords.get(lane, coords['left_open'])
    port_x = data['x']
    port_y = data['goal_y']

    # Init Nav2 — identical to simple_open_water.py
    nav = BasicNavigator()
    print(f'Initialising AMCL for {lane.upper()}...')
    nav.setInitialPose(make_pose(nav, data['x'], data['y'], 90.0))
    nav.waitUntilNav2Active(localizer='amcl')
    print('Nav2 ready.')

    helper = CargoHelper()

    # ── Phase 1: Drive to port (identical to simple_open_water.py) ──
    print('\n-- Phase 1: Drive to port --')
    nav.goToPose(make_pose(nav, port_x, port_y, 90.0))
    while not nav.isTaskComplete():
        fb = nav.getFeedback()
        if fb:
            dist = fb.distance_remaining
            print(f'  Distance to port: {dist:.2f} m', end='\r')
            if dist < 0.40:
                print('\n  Close enough — stopping Nav2')
                nav.cancelTask()
                break

    # ── Phase 2: Drop cargo ──────────────────────────────────────────
    print('\n-- Phase 2: Drop cargo --')
    helper.send_cargo_cmd(CMD_DROP, 'DROP (0x02)')
    helper.spin_pause(DROP_PAUSE)

    # ── Phase 3: Back up ─────────────────────────────────────────────
    print('\n-- Phase 3: Back up --')
    deadline = time.time() + BACKUP_TIME
    while time.time() < deadline:
        helper.drive(BACKUP_SPEED)
        rclpy.spin_once(helper, timeout_sec=0.05)
    helper.stop()
    time.sleep(0.3)

    # ── Phase 4: Maneuver to approach position ───────────────────────
    print('\n-- Phase 4: Maneuver to approach position --')
    approach_x = port_x + MANEUVER_X_OFFSET
    approach_y = port_y + MANEUVER_Y_OFFSET
    go_to(nav, helper, approach_x, approach_y, APPROACH_YAW_DEG,
          'approach position')

    # ── Phase 5: CV approach cargo ───────────────────────────────────
    print('\n-- Phase 5: CV approach --')
    helper.cv_approach_cargo()

    # ── Phase 6: Pick up cargo ───────────────────────────────────────
    print('\n-- Phase 6: Pick up cargo --')
    helper.send_cargo_cmd(CMD_PICKUP, 'PICKUP (0x01)')
    helper.spin_pause(PICKUP_PAUSE)
    helper.stop()

    # ── Phase 7: Return home ─────────────────────────────────────────
    print('\n-- Phase 7: Return home --')
    go_to(nav, helper, data['x'], data['y'], data['home_yaw'], 'home')

    print('\nRun complete.')
    helper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
