#!/usr/bin/env python3
"""
===================================================================
Open Waters Mapper — Teleop + Waypoint Marker for Open Waters Path
===================================================================
Drive the robot around the open waters arena with keyboard controls
while AMCL is running. Mark key positions for the navigator:
  - Spawn position (start)
  - FSK trigger point (drive-by blue panel)
  - Forward route waypoints (straight path to port)
  - Port zone corners and cargo positions
  - Lifeboat scan waypoints (return sweep positions)

The tool auto-detects left vs right open waters side by checking
which side the wall is on at startup (via lidar).

Controls:
  W/A/S/D  — Forward / Left / Back / Right
  Q/E      — Rotate left / Rotate right (in place)
  SPACE    — Emergency stop
  +/-      — Increase / decrease speed

Marking:
  0        — Mark spawn position
  1        — Mark FSK trigger waypoint (drive-by point)
  2-9      — Mark forward route waypoint #N
  G        — Mark generic route waypoint (appended to route list)
  P        — Mark port corner (marks 4 sequentially then wraps)
  C        — Mark cargo center position
  R        — Mark drop-off position (where to release starting cargo)
  B        — Mark maneuver/go-around waypoint (for port interaction)
  T        — Mark pickup approach position
  L        — Mark lifeboat scan position (return sweep points)

  I        — Print all marked positions
  V        — Save to JSON
  U        — Undo last mark
  F        — Load from JSON
  H        — Show help

  X / ESC  — Quit (auto-saves)

Requires:
  Terminal 1:  ros2 launch dalmotor robot.launch.py
  Terminal 2:  ros2 launch eced3901 arena_real_amcl.launch.py
  Terminal 3:  ros2 run eced3901 open_waters_mapper.py
"""

import json
import math
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String
import tf2_ros

# ── Speed presets ──
SPEED_PRESETS = [0.05, 0.08, 0.10, 0.13, 0.18, 0.22]
ROTATE_PRESETS = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
DEFAULT_SPEED_IDX = 2  # 0.10 m/s

# ── Lidar safety ──
LIDAR_FRONT_STOP = 0.12
LIDAR_FRONT_SLOW = 0.40

# ── File output ──
SAVE_DIR = os.path.expanduser('~/ros2_ws/src/eced3901/config')
SAVE_FILE_LEFT = os.path.join(SAVE_DIR, 'open_waters_waypoints_left.json')
SAVE_FILE_RIGHT = os.path.join(SAVE_DIR, 'open_waters_waypoints_right.json')

DT = 1.0 / 30.0

HELP_TEXT = """
╔══════════════════════════════════════════════════════════╗
║        OPEN WATERS MAPPER — Waypoint Marker              ║
╠══════════════════════════════════════════════════════════╣
║  Movement:                                               ║
║    W/A/S/D   Drive fwd/left/back/right                   ║
║    Q/E       Rotate left/right in place                  ║
║    SPACE     Emergency stop                              ║
║    +/-       Speed up / slow down                        ║
║                                                          ║
║  Marking:                                                ║
║    0         Mark SPAWN position (start)                 ║
║    1         Mark FSK trigger waypoint (drive-by)        ║
║    2-9       Mark forward route waypoint #N              ║
║    G         Mark generic route waypoint (sequential)    ║
║    P         Mark PORT corner (4 sequentially)           ║
║    C         Mark CARGO center                           ║
║    R         Mark DROP-OFF position (release cargo)      ║
║    B         Mark go-around/maneuver waypoint            ║
║    T         Mark PICKUP approach position               ║
║    L         Mark LIFEBOAT SCAN position (return sweep)  ║
║                                                          ║
║  Commands:                                               ║
║    I         Print all marks                             ║
║    V         Save to JSON                                ║
║    U         Undo last mark                              ║
║    F         Load from JSON                              ║
║    H         Show this help                              ║
║    X / ESC   Quit (auto-saves)                           ║
╚══════════════════════════════════════════════════════════╝
"""


class OpenWatersMapper(Node):
    def __init__(self):
        super().__init__('open_waters_mapper')

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/cv/nav_status', 10)

        self.last_odom = None
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.last_imu = None
        self.create_subscription(Imu, '/bno055/imu', self._imu_cb, 10)
        self.last_scan = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # State
        self.speed_idx = DEFAULT_SPEED_IDX
        self.marks = []
        self.waters_side = None  # 'left' or 'right'
        self.port_corner_count = 0
        self.maneuver_wp_count = 0
        self.route_wp_count = 0
        self.scan_wp_count = 0
        self.running = True

    def _odom_cb(self, msg):
        self.last_odom = msg

    def _imu_cb(self, msg):
        self.last_imu = msg

    def _scan_cb(self, msg):
        self.last_scan = msg

    # ── Pose ──
    def get_pose(self):
        """Get (x, y, yaw) from TF map->base_footprint, fallback to odom."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.15))
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)
            return (x, y, yaw)
        except Exception:
            pass
        if self.last_odom is not None:
            p = self.last_odom.pose.pose.position
            q = self.last_odom.pose.pose.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            return (p.x, p.y, math.atan2(siny, cosy))
        return None

    # ── Lidar ──
    def get_front_min(self):
        scan = self.last_scan
        if scan is None:
            return float('inf')
        n = len(scan.ranges)
        if n == 0:
            return float('inf')
        r_min = float('inf')
        for deg in list(range(0, 16)) + list(range(n - 15, n)):
            idx = deg % n
            r = scan.ranges[idx]
            if scan.range_min < r < scan.range_max:
                r_min = min(r_min, r)
        return r_min

    def detect_waters_side(self):
        """Detect left vs right open waters from lidar.
        In SLAM frame: +Y = west (robot's left at start).
        Left open waters: wall close on the +Y (left) side -> lidar ~90deg
        Right open waters: wall close on the -Y (right) side -> lidar ~270deg
        """
        scan = self.last_scan
        if scan is None:
            return None
        n = len(scan.ranges)
        if n == 0:
            return None

        def sector_min(start_deg, end_deg):
            r_min = float('inf')
            for deg in range(start_deg, end_deg + 1):
                idx = deg % n
                r = scan.ranges[idx]
                if scan.range_min < r < scan.range_max:
                    r_min = min(r_min, r)
            return r_min

        left_min = sector_min(60, 120)   # ~90deg = robot's left
        right_min = sector_min(240, 300)  # ~270deg = robot's right

        self.get_logger().info(
            f'Side detection: left_min={left_min:.2f}m, right_min={right_min:.2f}m')

        if left_min < right_min:
            return 'left'
        else:
            return 'right'

    # ── Status ──
    def set_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    # ── Markers ──
    def mark_position(self, label):
        rclpy.spin_once(self, timeout_sec=0.1)
        pose = self.get_pose()
        if pose is None:
            print(f'  ✗ Cannot mark — no pose available')
            return
        x, y, yaw = pose
        entry = {
            'label': label,
            'x': round(x, 4),
            'y': round(y, 4),
            'yaw': round(yaw, 4),
            'yaw_deg': round(math.degrees(yaw), 1),
            'timestamp': datetime.now().strftime('%H:%M:%S'),
        }
        self.marks.append(entry)
        num = sum(1 for m in self.marks if m['label'] == label)
        print(f'  ✓ Marked [{label} #{num}]: ({x:.3f}, {y:.3f}) yaw={math.degrees(yaw):.1f}°')
        self.set_status(f'Marked: {label} ({x:.2f},{y:.2f})')

    def undo_last(self):
        if not self.marks:
            print('  No marks to undo')
            return
        removed = self.marks.pop()
        # Adjust counters
        if removed['label'].startswith('port_corner_'):
            self.port_corner_count = max(0, self.port_corner_count - 1)
        elif removed['label'].startswith('maneuver_'):
            self.maneuver_wp_count = max(0, self.maneuver_wp_count - 1)
        elif removed['label'].startswith('route_wp_'):
            self.route_wp_count = max(0, self.route_wp_count - 1)
        elif removed['label'].startswith('scan_wp_'):
            self.scan_wp_count = max(0, self.scan_wp_count - 1)
        print(f'  ↩ Undid: {removed["label"]} at ({removed["x"]:.3f}, {removed["y"]:.3f})')

    def print_marks(self):
        if not self.marks:
            print('\n  No positions marked yet.\n')
            return
        print(f'\n  ══════ Marked Positions ({len(self.marks)}) ══════')
        print(f'  Open waters side: {self.waters_side or "not set"}')

        # Group by category
        categories = {}
        for m in self.marks:
            cat = m['label'].split('_')[0] if '_' in m['label'] else m['label']
            categories.setdefault(cat, []).append(m)

        for cat, items in categories.items():
            print(f'\n  --- {cat.upper()} ---')
            for m in items:
                print(f'    {m["label"]:20s}  '
                      f'({m["x"]:+7.3f}, {m["y"]:+7.3f})  '
                      f'yaw={m["yaw_deg"]:+6.1f}°  @ {m["timestamp"]}')
        print('  ═══════════════════════════════════════\n')

    def get_save_path(self):
        if self.waters_side == 'right':
            return SAVE_FILE_RIGHT
        return SAVE_FILE_LEFT

    def save_marks(self):
        os.makedirs(SAVE_DIR, exist_ok=True)
        fpath = self.get_save_path()
        data = {
            'description': f'Open waters waypoints — {self.waters_side or "unknown"} side',
            'waters_side': self.waters_side or 'left',
            'saved_at': datetime.now().isoformat(),
            'marks': self.marks,
        }

        # Build structured waypoints for the navigator
        structured = {}

        # Spawn
        spawns = [m for m in self.marks if m['label'] == 'spawn']
        if spawns:
            s = spawns[-1]
            structured['spawn'] = {'x': s['x'], 'y': s['y'], 'yaw': s['yaw']}

        # FSK trigger point
        fsk_marks = [m for m in self.marks if m['label'] == 'fsk_trigger']
        if fsk_marks:
            f = fsk_marks[-1]
            structured['fsk_trigger'] = {'x': f['x'], 'y': f['y'], 'yaw': f['yaw']}

        # Forward route: fwd_N marks (keys 2-9)
        fwd_marks = [m for m in self.marks if m['label'].startswith('fwd_')]
        if fwd_marks:
            structured['forward_route'] = [
                {'x': m['x'], 'y': m['y'], 'label': m['label']}
                for m in fwd_marks
            ]

        # Return route: route_wp_N marks (G key)
        ret_marks = [m for m in self.marks if m['label'].startswith('route_wp_')]
        if ret_marks:
            structured['return_route'] = [
                {'x': m['x'], 'y': m['y'], 'label': m['label']}
                for m in ret_marks
            ]

        # Lifeboat scan waypoints
        scan_marks = [m for m in self.marks if m['label'].startswith('scan_wp_')]
        if scan_marks:
            structured['scan_waypoints'] = [
                {'x': m['x'], 'y': m['y'], 'yaw': m['yaw'], 'label': m['label']}
                for m in scan_marks
            ]

        # Port corners
        port_corners = [m for m in self.marks if m['label'].startswith('port_corner_')]
        if port_corners:
            structured['port_corners'] = [
                {'x': m['x'], 'y': m['y']} for m in port_corners
            ]

        # Cargo
        cargo_marks = [m for m in self.marks if m['label'] == 'cargo_center']
        if cargo_marks:
            c = cargo_marks[-1]
            structured['cargo'] = {'x': c['x'], 'y': c['y']}

        # Drop-off
        drop_marks = [m for m in self.marks if m['label'] == 'drop_off']
        if drop_marks:
            d = drop_marks[-1]
            structured['drop_off'] = {'x': d['x'], 'y': d['y'], 'yaw': d['yaw']}

        # Maneuver waypoints
        man_marks = [m for m in self.marks if m['label'].startswith('maneuver_')]
        if man_marks:
            structured['maneuver_waypoints'] = [
                {'x': m['x'], 'y': m['y']} for m in man_marks
            ]

        # Pickup approach
        pickup_marks = [m for m in self.marks if m['label'] == 'pickup_approach']
        if pickup_marks:
            p = pickup_marks[-1]
            structured['pickup_approach'] = {'x': p['x'], 'y': p['y'], 'yaw': p['yaw']}

        data['waypoints'] = structured

        with open(fpath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'  💾 Saved {len(self.marks)} marks to {fpath}')

    def load_marks(self):
        fpath = self.get_save_path()
        if not os.path.exists(fpath):
            # Try the other side
            alt = SAVE_FILE_RIGHT if fpath == SAVE_FILE_LEFT else SAVE_FILE_LEFT
            if os.path.exists(alt):
                fpath = alt
            else:
                print(f'  No saved waypoints found')
                return
        with open(fpath, 'r') as f:
            data = json.load(f)
        self.marks = data.get('marks', [])
        self.waters_side = data.get('waters_side', self.waters_side)
        # Restore counters
        self.port_corner_count = sum(
            1 for m in self.marks if m['label'].startswith('port_corner_'))
        self.maneuver_wp_count = sum(
            1 for m in self.marks if m['label'].startswith('maneuver_'))
        self.route_wp_count = sum(
            1 for m in self.marks if m['label'].startswith('route_wp_'))
        self.scan_wp_count = sum(
            1 for m in self.marks if m['label'].startswith('scan_wp_'))
        print(f'  📂 Loaded {len(self.marks)} marks from {fpath}')
        print(f'     Waters side: {self.waters_side}')


def get_key(timeout=0.05):
    """Non-blocking key read from terminal (Linux)."""
    if select.select([sys.stdin], [], [], timeout)[0]:
        return sys.stdin.read(1)
    return None


def main():
    rclpy.init()
    node = OpenWatersMapper()

    # Wait for pose
    print('Waiting for robot pose...')
    t0 = time.time()
    while time.time() - t0 < 15.0:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.get_pose() is not None:
            break
    pose = node.get_pose()
    if pose is None:
        print('ERROR: No pose available. Is robot.launch.py + AMCL running?')
        node.destroy_node()
        rclpy.shutdown()
        return

    print(f'Pose OK: ({pose[0]:.3f}, {pose[1]:.3f}) yaw={math.degrees(pose[2]):.1f}°')

    # Auto-detect open waters side
    print('Detecting open waters side from lidar...')
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.2)
    node.waters_side = node.detect_waters_side()
    if node.waters_side:
        print(f'  Detected: {node.waters_side.upper()} open waters')
    else:
        print('  Could not auto-detect — defaulting to LEFT open waters')
        node.waters_side = 'left'

    # Try loading existing waypoints
    node.load_marks()

    print(HELP_TEXT)

    # Set terminal to raw mode for keyboard input
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        _run_loop(node)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        # Auto-save on exit
        if node.marks:
            print('\nAuto-saving marks...')
            node.save_marks()
        # Stop robot
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


def _run_loop(node):
    cmd = Twist()
    last_print = 0

    while node.running and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)

        key = get_key(0.03)
        if key is None:
            # Smooth deceleration
            cmd.linear.x *= 0.85
            cmd.angular.z *= 0.85
            if abs(cmd.linear.x) < 0.01:
                cmd.linear.x = 0.0
            if abs(cmd.angular.z) < 0.01:
                cmd.angular.z = 0.0
        else:
            key = key.lower()
            speed = SPEED_PRESETS[node.speed_idx]
            rot = ROTATE_PRESETS[node.speed_idx]

            # ── Movement ──
            if key == 'w':
                cmd.linear.x = speed
                cmd.angular.z = 0.0
            elif key == 's':
                cmd.linear.x = -speed
                cmd.angular.z = 0.0
            elif key == 'a':
                cmd.linear.x = speed * 0.5
                cmd.angular.z = rot
            elif key == 'd':
                cmd.linear.x = speed * 0.5
                cmd.angular.z = -rot
            elif key == 'q':
                cmd.linear.x = 0.0
                cmd.angular.z = rot
            elif key == 'e':
                cmd.linear.x = 0.0
                cmd.angular.z = -rot
            elif key == ' ':
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

            # ── Speed control ──
            elif key == '+' or key == '=':
                node.speed_idx = min(node.speed_idx + 1, len(SPEED_PRESETS) - 1)
                print(f'  Speed: {SPEED_PRESETS[node.speed_idx]:.2f} m/s  '
                      f'Rotate: {ROTATE_PRESETS[node.speed_idx]:.2f} rad/s')
            elif key == '-' or key == '_':
                node.speed_idx = max(node.speed_idx - 1, 0)
                print(f'  Speed: {SPEED_PRESETS[node.speed_idx]:.2f} m/s  '
                      f'Rotate: {ROTATE_PRESETS[node.speed_idx]:.2f} rad/s')

            # ── Marking ──
            elif key == '0':
                node.mark_position('spawn')
            elif key == '1':
                node.mark_position('fsk_trigger')
            elif key in '23456789':
                node.mark_position(f'fwd_{key}')
            elif key == 'g':
                node.route_wp_count += 1
                node.mark_position(f'route_wp_{node.route_wp_count}')
            elif key == 'p':
                node.port_corner_count += 1
                corner_num = ((node.port_corner_count - 1) % 4) + 1
                node.mark_position(f'port_corner_{corner_num}')
            elif key == 'c':
                node.mark_position('cargo_center')
            elif key == 'r':
                node.mark_position('drop_off')
            elif key == 'b':
                node.maneuver_wp_count += 1
                node.mark_position(f'maneuver_{node.maneuver_wp_count}')
            elif key == 't':
                node.mark_position('pickup_approach')
            elif key == 'l':
                node.scan_wp_count += 1
                node.mark_position(f'scan_wp_{node.scan_wp_count}')

            # ── Commands ──
            elif key == 'i':
                node.print_marks()
            elif key == 'v':
                node.save_marks()
            elif key == 'u':
                node.undo_last()
            elif key == 'f':
                node.load_marks()
            elif key == 'h':
                print(HELP_TEXT)
            elif key == 'x' or key == '\x1b':  # x or ESC
                print('\nQuitting...')
                node.running = False
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0

        # Lidar safety
        front_min = node.get_front_min()
        if cmd.linear.x > 0:
            if front_min < LIDAR_FRONT_STOP:
                cmd.linear.x = 0.0
            elif front_min < LIDAR_FRONT_SLOW:
                cmd.linear.x *= max(0.3, front_min / LIDAR_FRONT_SLOW)

        node.cmd_pub.publish(cmd)

        # Status line
        now = time.time()
        if now - last_print > 0.5:
            last_print = now
            pose = node.get_pose()
            spd = SPEED_PRESETS[node.speed_idx]
            side = (node.waters_side or '?')[0].upper()
            if pose:
                x, y, yaw = pose
                status = (f'[{side}] Pos: ({x:+6.3f},{y:+6.3f}) '
                          f'yaw={math.degrees(yaw):+6.1f}° '
                          f'| Speed: {spd:.2f} | Marks: {len(node.marks)} '
                          f'| Front: {front_min:.2f}m')
                sys.stdout.write(f'\r{status}    ')
                sys.stdout.flush()

    # Final stop
    node.cmd_pub.publish(Twist())


if __name__ == '__main__':
    main()
