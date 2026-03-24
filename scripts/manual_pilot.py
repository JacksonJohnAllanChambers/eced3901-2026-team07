#!/usr/bin/env python3
"""
manual_pilot.py  —  Teleoperate the robot through the course with live
lifeboat triangulation and CV data logging.

After the run, prints:
  • Lifeboat triangulator estimate vs. marked ground truth
  • Cargo CV detection log (pose + bearing + distance at each detection)
  • Port approach quality summary

Controls
────────────────────────────────────────────────────────────────
  W / S          forward / backward
  A / D          turn left / right (hold to spin)
  SPACE          stop
  P              PICKUP cargo  →  serial 0x01
  X              DROP cargo    →  serial 0x02
  L              mark LIFEBOAT ground truth at current pose
  C              mark CARGO ground truth at current pose
  T              record trajectory waypoint
  + / -          increase / decrease linear speed
  ?              print current status
  Q  or  Ctrl-C  quit & print summary
────────────────────────────────────────────────────────────────

Usage
  ros2 run eced3901 manual_pilot.py --side left
  ros2 run eced3901 manual_pilot.py --side right
  ros2 run eced3901 manual_pilot.py --side left --serial /dev/ttyUSB1
"""

import argparse
import json
import math
import os
import queue
import sys
import termios
import threading
import time
import tty

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import tf2_ros
from tf2_ros import TransformException

# ── Local import: lifeboat triangulator ────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
from lifeboat_triangulator import (
    LifeboatTriangulator,
    LEFT_X_MIN, LEFT_X_MAX,
    RIGHT_X_MIN, RIGHT_X_MAX,
)

# ── Movement constants ──────────────────────────────────────────
DEFAULT_LINEAR  = 0.12     # m/s
DEFAULT_ANGULAR = 0.55     # rad/s
SPEED_STEP      = 0.02
MAX_LINEAR      = 0.25
MIN_LINEAR      = 0.04

# ── CV ──────────────────────────────────────────────────────────
CV_MIN_CONFIDENCE = 0.30
PORT_AREA_Y       = 2.50   # Y threshold: above this = "near port"

# ── Serial ──────────────────────────────────────────────────────
CMD_PICKUP = b'\x01'
CMD_DROP   = b'\x02'


# ═══════════════════════════════════════════════════════════════
# Keyboard helper  (runs in its own thread)
# ═══════════════════════════════════════════════════════════════
def _keyboard_thread(key_queue: queue.Queue, stop_event: threading.Event):
    """Read single chars from stdin in raw mode, push to key_queue."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not stop_event.is_set():
            ch = sys.stdin.read(1)
            key_queue.put(ch)
            if ch in ('q', 'Q', '\x03'):   # quit / Ctrl-C
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ═══════════════════════════════════════════════════════════════
# Pilot Node
# ═══════════════════════════════════════════════════════════════
class ManualPilot(Node):

    def __init__(self, side: str, serial_dev: str):
        super().__init__('manual_pilot')
        self.side = side

        # ── Publishers ──────────────────────────────────────────
        self.cmd_pub    = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/cv/nav_status', 10)

        # ── Subscribers ─────────────────────────────────────────
        self.last_odom        = None
        self.last_detections  = []
        self.last_det_time    = 0.0
        self.last_scan        = None
        self.create_subscription(Odometry, '/odom',          self._odom_cb,  10)
        self.create_subscription(String,   '/cv/detections', self._det_cb,   10)
        self.create_subscription(LaserScan, '/scan',         self._scan_cb,  10)

        # ── TF for AMCL pose ────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── Serial port (cargo) ──────────────────────────────────
        self._serial = None
        self._init_serial(serial_dev)

        # ── Lifeboat triangulator ───────────────────────────────
        if side == 'right':
            self.triangulator = LifeboatTriangulator(
                valid_x_min=RIGHT_X_MIN, valid_x_max=RIGHT_X_MAX)
        else:
            self.triangulator = LifeboatTriangulator(
                valid_x_min=LEFT_X_MIN,  valid_x_max=LEFT_X_MAX)

        # ── Run-time state ──────────────────────────────────────
        self.linear_speed    = DEFAULT_LINEAR
        self.angular_speed   = DEFAULT_ANGULAR
        self.current_cmd     = (0.0, 0.0)   # (linear, angular)

        # ── Logging ─────────────────────────────────────────────
        self.trajectory     = []            # [(t, x, y, yaw), ...]
        self.cargo_log      = []            # cargo CV detections + pose
        self.lifeboat_log   = []            # all lifeboat detections + pose
        self.lifeboat_truth = None          # (x, y) if 'L' pressed
        self.cargo_truth    = None          # (x, y) if 'C' pressed
        self.events         = []            # [(t, text), ...] named events
        self.start_time     = time.time()
        self._last_pose_log  = 0.0          # throttle trajectory logging

    # ── Callbacks ───────────────────────────────────────────────

    def _odom_cb(self, msg):
        self.last_odom = msg

    def _scan_cb(self, msg):
        self.last_scan = msg

    def _det_cb(self, msg):
        try:
            dets = json.loads(msg.data)
            self.last_detections = dets
            self.last_det_time   = time.time()

            pose = self._get_pose()
            if pose is None:
                return
            rx, ry, ryaw = pose

            for d in dets:
                if d.get('confidence', 0) < CV_MIN_CONFIDENCE:
                    continue
                label = d.get('label', '')

                # ── feed triangulator for lifeboat ───────────────
                if label == 'lifeboat':
                    self.triangulator.add_observation(
                        rx, ry, ryaw,
                        d.get('bearing_deg', 0.0),
                        d.get('confidence', 0.5),
                        d.get('ground_distance_cm', 100.0),
                    )
                    entry = {
                        't':           round(time.time() - self.start_time, 2),
                        'robot':       (round(rx, 3), round(ry, 3)),
                        'robot_yaw':   round(math.degrees(ryaw), 1),
                        'bearing_deg': round(d.get('bearing_deg', 0), 2),
                        'dist_cm':     round(d.get('ground_distance_cm', 0), 1),
                        'conf':        round(d.get('confidence', 0), 3),
                        'area':        round(d.get('area', 0), 0),
                    }
                    self.lifeboat_log.append(entry)

                # ── log cargo detections ─────────────────────────
                elif label == 'cargo':
                    entry = {
                        't':           round(time.time() - self.start_time, 2),
                        'robot':       (round(rx, 3), round(ry, 3)),
                        'robot_yaw':   round(math.degrees(ryaw), 1),
                        'bearing_deg': round(d.get('bearing_deg', 0), 2),
                        'dist_cm':     round(d.get('ground_distance_cm', 0), 1),
                        'conf':        round(d.get('confidence', 0), 3),
                        'area':        round(d.get('area', 0), 0),
                    }
                    self.cargo_log.append(entry)

        except (json.JSONDecodeError, Exception):
            self.last_detections = []

    # ── Pose helper ─────────────────────────────────────────────

    def _get_pose(self):
        """Return (x, y, yaw) from TF map→base_footprint, or None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', 'base_footprint',
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.05))
            t = tf.transform.translation
            q = tf.transform.rotation
            yaw = math.atan2(
                2*(q.w*q.z + q.x*q.y),
                1 - 2*(q.y*q.y + q.z*q.z))
            return float(t.x), float(t.y), float(yaw)
        except TransformException:
            pass

        # Fallback: odometry (will drift, but better than nothing)
        if self.last_odom is not None:
            p = self.last_odom.pose.pose
            q = p.orientation
            yaw = math.atan2(
                2*(q.w*q.z + q.x*q.y),
                1 - 2*(q.y*q.y + q.z*q.z))
            return float(p.position.x), float(p.position.y), float(yaw)
        return None

    # ── Serial ──────────────────────────────────────────────────

    def _init_serial(self, device: str):
        """Open serial port for cargo commands."""
        try:
            import serial as _serial
            self._serial = _serial.Serial(
                device, baudrate=9600,
                bytesize=8, parity='N', stopbits=1, timeout=0.1)
            self.get_logger().info(f'Serial port {device} open')
        except Exception as e:
            self.get_logger().warn(f'Serial open failed: {e} — cargo commands disabled')
            self._serial = None

    def _send_serial(self, data: bytes, label: str):
        """Send bytes to serial; log the event."""
        pose = self._get_pose()
        pos_str = f'({pose[0]:.3f},{pose[1]:.3f})' if pose else '(?)'
        if self._serial:
            try:
                self._serial.write(data)
                msg = f'SERIAL {label}  at {pos_str}'
                self.get_logger().info(msg)
                self.events.append((time.time() - self.start_time, msg))
                print(f'\n  ★  {msg}')
            except Exception as e:
                self.get_logger().error(f'Serial write failed: {e}')
        else:
            msg = f'SERIAL {label} (no port)  at {pos_str}'
            print(f'\n  ✗  {msg}')
            self.events.append((time.time() - self.start_time, msg))

    # ── Drive ────────────────────────────────────────────────────

    def _publish_cmd(self, lin: float, ang: float):
        self.current_cmd = (lin, ang)
        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        self.cmd_pub.publish(cmd)

    def stop(self):
        self._publish_cmd(0.0, 0.0)

    # ── Action handlers ─────────────────────────────────────────

    def handle_key(self, key: str) -> bool:
        """Process one keypress. Returns False if quit requested."""
        k = key.lower()

        if k in ('w',):
            self._publish_cmd(self.linear_speed, 0.0)
        elif k in ('s',):
            self._publish_cmd(-self.linear_speed, 0.0)
        elif k in ('a',):
            self._publish_cmd(0.0,  self.angular_speed)
        elif k in ('d',):
            self._publish_cmd(0.0, -self.angular_speed)
        elif k == ' ':
            self.stop()
            print('\n  [STOP]')
        elif k == 'p':
            self.stop()
            self._send_serial(CMD_PICKUP, 'PICKUP (0x01)')
        elif k == 'x':
            self.stop()
            self._send_serial(CMD_DROP, 'DROP (0x02)')
        elif k == 'l':
            self._mark_lifeboat_truth()
        elif k == 'c':
            self._mark_cargo_truth()
        elif k == 't':
            self._record_waypoint()
        elif key in ('+', '='):
            self.linear_speed = min(MAX_LINEAR, self.linear_speed + SPEED_STEP)
            print(f'\n  Speed → {self.linear_speed:.2f} m/s')
        elif key == '-':
            self.linear_speed = max(MIN_LINEAR, self.linear_speed - SPEED_STEP)
            print(f'\n  Speed → {self.linear_speed:.2f} m/s')
        elif k == '?':
            self._print_status()
        elif k in ('q', '\x03'):
            self.stop()
            return False

        return True

    def _mark_lifeboat_truth(self):
        pose = self._get_pose()
        if pose:
            self.lifeboat_truth = (pose[0], pose[1])
            msg = f'LIFEBOAT GROUND TRUTH marked at ({pose[0]:.3f},{pose[1]:.3f})'
        else:
            msg = 'LIFEBOAT GROUND TRUTH — no pose available!'
        self.events.append((time.time() - self.start_time, msg))
        print(f'\n  ★  {msg}')

    def _mark_cargo_truth(self):
        pose = self._get_pose()
        if pose:
            self.cargo_truth = (pose[0], pose[1])
            msg = f'CARGO GROUND TRUTH marked at ({pose[0]:.3f},{pose[1]:.3f})'
        else:
            msg = 'CARGO GROUND TRUTH — no pose available!'
        self.events.append((time.time() - self.start_time, msg))
        print(f'\n  ★  {msg}')

    def _record_waypoint(self):
        pose = self._get_pose()
        if pose:
            self.trajectory.append({
                't':   round(time.time() - self.start_time, 2),
                'x':   round(pose[0], 3),
                'y':   round(pose[1], 3),
                'yaw': round(math.degrees(pose[2]), 1),
                'manual': True,
            })
            print(f'\n  ▸  Waypoint T{len(self.trajectory)}: ({pose[0]:.3f},{pose[1]:.3f}) yaw={math.degrees(pose[2]):.1f}°')

    def _print_status(self):
        pose = self._get_pose()
        pos = f'({pose[0]:.3f},{pose[1]:.3f}) yaw={math.degrees(pose[2]):.1f}°' \
              if pose else '(no pose)'
        recent = [d for d in self.last_detections
                  if time.time() - self.last_det_time < 1.0]
        lifeboats = [d for d in recent if d.get('label') == 'lifeboat']
        cargos    = [d for d in recent if d.get('label') == 'cargo']
        lbest = self.triangulator.get_best_estimate()
        print(f"""
┌─ STATUS ──────────────────────────────────────────────────────────
│ Side: {self.side.upper()}   Speed: {self.linear_speed:.2f} m/s
│ Pose: {pos}
│ Detections: {len(lifeboats)} lifeboat, {len(cargos)} cargo  (det age {time.time()-self.last_det_time:.1f}s)
│ Triangulator: {self.triangulator.observation_count} observations
│   Best estimate: {f'({lbest[0]:.3f},{lbest[1]:.3f})' if lbest else 'insufficient data'}
│   Ground truth: {f'({self.lifeboat_truth[0]:.3f},{self.lifeboat_truth[1]:.3f})' if self.lifeboat_truth else 'not marked — press L'}
│ Cargo GT: {f'({self.cargo_truth[0]:.3f},{self.cargo_truth[1]:.3f})' if self.cargo_truth else 'not marked — press C'}
│ Cargo detections logged: {len(self.cargo_log)}
└───────────────────────────────────────────────────────────────────""")

    # ── Continuous pose logging ─────────────────────────────────

    def log_pose_if_due(self):
        """Log pose every 0.5s to trajectory."""
        now = time.time()
        if now - self._last_pose_log < 0.5:
            return
        self._last_pose_log = now
        pose = self._get_pose()
        if pose:
            self.trajectory.append({
                't':   round(now - self.start_time, 2),
                'x':   round(pose[0], 3),
                'y':   round(pose[1], 3),
                'yaw': round(math.degrees(pose[2]), 1),
            })

    # ── Final summary ────────────────────────────────────────────

    def print_summary(self):
        duration = time.time() - self.start_time
        sep = '═' * 66
        print(f'\n{sep}')
        print(f'  MANUAL PILOT RUN SUMMARY  — side={self.side.upper()}  duration={duration:.1f}s')
        print(sep)

        # ── Events ────────────────────────────────────────────
        if self.events:
            print('\n── Events ──────────────────────────────────────────────────')
            for t, msg in self.events:
                print(f'  [{t:6.1f}s]  {msg}')

        # ── Lifeboat triangulator ─────────────────────────────
        print('\n── Lifeboat Triangulator ───────────────────────────────────')
        lbest = self.triangulator.get_best_estimate()
        print(f'  Observations accumulated: {self.triangulator.observation_count}')
        if lbest:
            print(f'  Best estimate: ({lbest[0]:.3f}, {lbest[1]:.3f})')
        else:
            print(f'  Best estimate: None (need ≥ 2 observations with ≥ 2 rays in spawn zone)')
        if self.lifeboat_truth:
            gt = self.lifeboat_truth
            print(f'  Ground truth:  ({gt[0]:.3f}, {gt[1]:.3f})')
            if lbest:
                err = math.hypot(lbest[0] - gt[0], lbest[1] - gt[1]) * 100
                status = '✓ GOOD' if err < 20 else ('~ OK' if err < 40 else '✗ POOR')
                print(f'  Error: {err:.1f} cm  [{status}]')
            else:
                print('  Cannot compute error — no estimate yet')
        else:
            print('  Ground truth: NOT MARKED (drive to lifeboat and press L)')

        print(f'\n  Lifeboat detection events (all {len(self.lifeboat_log)}, showing last 10):')
        for e in self.lifeboat_log[-10:]:
            print(f'    t={e["t"]:.1f}s  robot={e["robot"]}  yaw={e["robot_yaw"]}°'
                  f'  bearing={e["bearing_deg"]:+.1f}°  dist={e["dist_cm"]:.0f}cm'
                  f'  conf={e["conf"]:.2f}')

        # ── Cargo detection log ───────────────────────────────
        print('\n── Cargo Detections (approach alignment data) ──────────────')
        if self.cargo_log:
            print(f'  Total cargo detections logged: {len(self.cargo_log)}')
            if self.cargo_truth:
                ct = self.cargo_truth
                print(f'  Cargo ground truth: ({ct[0]:.3f}, {ct[1]:.3f})')
                print()
                print(f'  {"t(s)":>6}  {"robot_x":>8}  {"robot_y":>8}  {"yaw°":>7}'
                      f'  {"bear°":>7}  {"dist_cm":>8}  {"err_cm":>7}  {"area":>8}')
                print(f'  {"─"*6}  {"─"*8}  {"─"*8}  {"─"*7}'
                      f'  {"─"*7}  {"─"*8}  {"─"*7}  {"─"*8}')
                for e in self.cargo_log:
                    rx, ry = e['robot']
                    ryaw_r = math.radians(e['robot_yaw'])
                    # Predicted cargo position from bearing + distance
                    abs_bear = ryaw_r - math.radians(e['bearing_deg'])
                    dist_m   = e['dist_cm'] / 100.0
                    pred_x   = rx + dist_m * math.cos(abs_bear)
                    pred_y   = ry + dist_m * math.sin(abs_bear)
                    pred_err = math.hypot(pred_x - ct[0], pred_y - ct[1]) * 100
                    print(f'  {e["t"]:6.1f}  {rx:8.3f}  {ry:8.3f}  {e["robot_yaw"]:7.1f}'
                          f'  {e["bearing_deg"]:+7.1f}  {e["dist_cm"]:8.1f}'
                          f'  {pred_err:7.1f}  {e["area"]:8.0f}')
                # Recommend approach
                close = [e for e in self.cargo_log if e['dist_cm'] < 40]
                if close:
                    avg_bear = sum(e['bearing_deg'] for e in close) / len(close)
                    print(f'\n  Close detections (<40cm): {len(close)} readings')
                    print(f'  Avg bearing at close range: {avg_bear:+.1f}°')
                    if abs(avg_bear) > 15:
                        print(f'  ⚠  Large close bearing offset — robot is arriving')
                        print(f'     off-axis by ~{avg_bear:+.1f}°. Adjust approach waypoint.')
                    else:
                        print(f'  ✓  Close bearing near centre — approach angle looks good.')
            else:
                print(f'  No cargo ground truth marked (press C while touching cargo).')
                print(f'  Showing raw detection log ({len(self.cargo_log)} entries):')
                print(f'\n  {"t(s)":>6}  {"robot_x":>8}  {"robot_y":>8}  {"yaw°":>7}'
                      f'  {"bear°":>7}  {"dist_cm":>8}  {"area":>8}')
                print(f'  {"─"*6}  {"─"*8}  {"─"*8}  {"─"*7}'
                      f'  {"─"*7}  {"─"*8}  {"─"*8}')
                for e in self.cargo_log[-30:]:
                    rx, ry = e['robot']
                    print(f'  {e["t"]:6.1f}  {rx:8.3f}  {ry:8.3f}  {e["robot_yaw"]:7.1f}'
                          f'  {e["bearing_deg"]:+7.1f}  {e["dist_cm"]:8.1f}  {e["area"]:8.0f}')
        else:
            print('  No cargo detections during run.')

        # ── Trajectory summary ───────────────────────────────
        print('\n── Trajectory ──────────────────────────────────────────────')
        print(f'  Pose samples logged: {len(self.trajectory)}')
        if self.trajectory:
            xs  = [p['x']   for p in self.trajectory]
            ys  = [p['y']   for p in self.trajectory]
            mwp = [p for p in self.trajectory if p.get('manual')]
            print(f'  X range: {min(xs):.3f} → {max(xs):.3f}')
            print(f'  Y range: {min(ys):.3f} → {max(ys):.3f}')
            if mwp:
                print(f'  Manual waypoints recorded (T key):')
                for p in mwp:
                    print(f'    t={p["t"]:.1f}s  ({p["x"]:.3f},{p["y"]:.3f}) yaw={p["yaw"]:.1f}°')

        # ── Save JSON ────────────────────────────────────────
        out = {
            'side':             self.side,
            'duration_s':       round(duration, 1),
            'events':           self.events,
            'lifeboat_truth':   self.lifeboat_truth,
            'lifeboat_estimate': list(lbest) if lbest else None,
            'lifeboat_log':     self.lifeboat_log,
            'cargo_truth':      self.cargo_truth,
            'cargo_log':        self.cargo_log,
            'trajectory':       self.trajectory,
        }
        ts = time.strftime('%Y%m%d_%H%M%S')
        outfile = os.path.expanduser(
            f'~/ros2_ws/src/eced3901/scripts/pilot_{ts}.json')
        with open(outfile, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\n  Full data saved → {outfile}')
        print(sep)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Manual pilot with diagnostics')
    parser.add_argument('--side', choices=['left', 'right'], default='left',
                        help='Open waters side (default: left)')
    parser.add_argument('--serial', default='/dev/ttyUSB0',
                        help='Serial device for cargo (default: /dev/ttyUSB0)')
    args = parser.parse_args()

    rclpy.init()
    node = ManualPilot(side=args.side, serial_dev=args.serial)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          MANUAL PILOT  —  side={args.side.upper():<5}  serial={args.serial:<15}  ║
╠══════════════════════════════════════════════════════════════╣
║  W/S  forward/back    A/D  turn left/right    SPACE  stop   ║
║  P    pickup cargo    X    drop cargo                        ║
║  L    mark LIFEBOAT ground truth here                        ║
║  C    mark CARGO ground truth here                           ║
║  T    record trajectory waypoint                             ║
║  +/-  change speed    ?    status    Q  quit+summary         ║
╚══════════════════════════════════════════════════════════════╝
  Waiting for TF / CV pipeline...
""")

    # Warm up ROS2 subscriptions
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)

    # Start keyboard thread
    key_queue  = queue.Queue()
    stop_event = threading.Event()
    kb_thread  = threading.Thread(
        target=_keyboard_thread, args=(key_queue, stop_event), daemon=True)
    kb_thread.start()

    print('  Ready — start driving!\n')

    # ── Main loop ───────────────────────────────────────────────
    running = True
    _status_timer = time.time()
    try:
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            node.log_pose_if_due()

            # Process all pending keystrokes
            while not key_queue.empty():
                key = key_queue.get_nowait()
                running = node.handle_key(key)
                if not running:
                    break

            # Periodic mini-status line (every 5 s)
            now = time.time()
            if now - _status_timer > 5.0:
                _status_timer = now
                pose = node._get_pose()
                lbest = node.triangulator.get_best_estimate()
                pos_str = f'({pose[0]:.2f},{pose[1]:.2f})' if pose else '(?)'
                tri_str = (f'({lbest[0]:.2f},{lbest[1]:.2f})' if lbest
                           else f'{node.triangulator.observation_count}obs')
                print(f'  pos={pos_str}  tri={tri_str}'
                      f'  cargo_dets={len(node.cargo_log)}'
                      f'  spd={node.linear_speed:.2f}m/s', flush=True)

    except Exception as e:
        node.get_logger().error(f'Error in main loop: {e}')
    finally:
        stop_event.set()
        node.stop()

    # Print summary AFTER terminal is restored (keyboard thread returns first)
    kb_thread.join(timeout=1.0)
    node.print_summary()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
