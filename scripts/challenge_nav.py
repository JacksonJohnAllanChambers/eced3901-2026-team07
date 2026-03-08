#!/usr/bin/env python3
"""
===================================================================
Challenge Navigator — Optimized Coastal Challenge Run
===================================================================
Fresh rewrite of the coastal challenge navigator with:
  - LiDAR-based auto-detection of left vs right coastal path
  - Lifeboat detection on BOTH forward and return paths (10s pause)
  - No excessive 360° scanning
  - Proper port interaction: drop cargo → maneuver → pickup target

Phases:
  A. DETECT   — Auto-detect left/right coastal via lidar
  B. FORWARD  — Zigzag to port with lifeboat scanning
  C. PORT     — Drop starting cargo, maneuver, pickup target cargo
  D. RETURN   — Zigzag home with lifeboat scanning
  E. DONE     — Stop and report

Usage:
  1. ros2 launch dalmotor robot.launch.py
  2. ros2 launch eced3901 arena_real_amcl.launch.py
  3. ros2 launch dalibot_cv dalibot_cv_launch.py
  4. ros2 run eced3901 challenge_nav.py

Optionally use challenge_mapper.py waypoints:
  ros2 run eced3901 challenge_nav.py --waypoints ~/ros2_ws/src/eced3901/config/challenge_waypoints_left.json
"""

import json
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String, Bool
import tf2_ros

# ─────────────────────────────────────────────────────────────────
# SLAM map coordinate constants
# +X = arena NORTH (robot forward at start)
# +Y = arena WEST  (robot left at start)
# ─────────────────────────────────────────────────────────────────
NORTH = 0.0
SOUTH = math.pi
EAST  = -math.pi / 2
WEST  = math.pi / 2

# ── Default coordinates (left coastal) ──
LEFT_START_X   = 0.0
LEFT_START_Y   = 0.0
LEFT_START_YAW = 0.0

LEFT_WALL_X = (0.85, 1.80, 2.70)
LEFT_GAP_Y  = (0.054, 0.653, 0.116)
LEFT_MID_X_12 = (LEFT_WALL_X[0] + LEFT_WALL_X[1]) / 2
LEFT_MID_X_23 = (LEFT_WALL_X[1] + LEFT_WALL_X[2]) / 2
LEFT_CORRIDOR_Y = 0.35
LEFT_PORT_X = 3.40
LEFT_PORT_Y = 0.0

# ── Right coastal (mirrored: Y-values negated) ──
RIGHT_START_X   = 0.0
RIGHT_START_Y   = 0.0
RIGHT_START_YAW = 0.0

RIGHT_WALL_X = (0.85, 1.80, 2.70)
RIGHT_GAP_Y  = (-0.054, -0.653, -0.116)
RIGHT_MID_X_12 = (RIGHT_WALL_X[0] + RIGHT_WALL_X[1]) / 2
RIGHT_MID_X_23 = (RIGHT_WALL_X[1] + RIGHT_WALL_X[2]) / 2
RIGHT_CORRIDOR_Y = -0.35
RIGHT_PORT_X = 3.40
RIGHT_PORT_Y = 0.0

# ── Speeds ──
LINEAR_SPEED  = 0.13
ROTATE_SPEED  = 0.30
CV_LINEAR     = 0.10
CV_ROTATE     = 0.20

# ── Tolerances ──
YAW_TOL       = 0.05
DIST_TOL      = 0.15

# ── Steering ──
STEER_GAIN     = 1.2
MAX_STEER      = 0.5
STEER_DEADBAND = 0.03
STUCK_TIME     = 5.0
GOTO_TIMEOUT   = 25.0

# ── Lidar ──
LIDAR_SIDE_WARN  = 0.13
LIDAR_SIDE_GAIN  = 3.0
LIDAR_SLOW_DIST  = 0.55
LIDAR_FRONT_STOP = 0.12

# ── CV detection thresholds ──
CV_CENTER_X           = 640
CV_CENTER_TOL         = 80
CV_CARGO_AREA_CLOSE   = 35000
CV_LIFEBOAT_AREA_CLOSE = 8000
CV_MIN_CONFIDENCE     = 0.35
CV_SCAN_TIMEOUT       = 15.0
CV_APPROACH_TIMEOUT   = 20.0
CAM_WIDTH             = 1280
CAM_HEIGHT            = 720
CV_BOTTOM_Y           = 620

# ── Lifeboat false-positive filtering ──
# Lifeboat detections with area above this are likely cargo misidentified
LIFEBOAT_MAX_AREA     = 12000   # real lifeboat is small from distance
# Minimum area to consider a lifeboat detection valid (too tiny = noise)
LIFEBOAT_MIN_AREA     = 300
# Don't react to lifeboat detections when robot is past wall 3 (near port)
# because cargo at port gets misidentified as lifeboat
LIFEBOAT_MAX_X        = 2.80    # SLAM x beyond which we suppress lifeboat

# ── Timing ──
LIFEBOAT_PAUSE_SECONDS = 10.0
CARGO_DROP_PAUSE       = 3.0
CARGO_PICKUP_PAUSE     = 5.0

DT = 1.0 / 30.0


# ─────────────────────────────────────────────────────────────────
# Route builders
# ─────────────────────────────────────────────────────────────────
def build_forward_route(side='left'):
    """Build forward zigzag route through the 3 walls to the port."""
    if side == 'left':
        return [
            (LEFT_WALL_X[0], LEFT_GAP_Y[0], 'north'),
            (LEFT_MID_X_12,  LEFT_CORRIDOR_Y),
            (LEFT_WALL_X[1], LEFT_GAP_Y[1], 'north'),
            (LEFT_MID_X_23,  LEFT_CORRIDOR_Y),
            (LEFT_WALL_X[2], LEFT_GAP_Y[2], 'north'),
            (LEFT_PORT_X,    LEFT_PORT_Y),
        ]
    else:
        return [
            (RIGHT_WALL_X[0], RIGHT_GAP_Y[0], 'north'),
            (RIGHT_MID_X_12,  RIGHT_CORRIDOR_Y),
            (RIGHT_WALL_X[1], RIGHT_GAP_Y[1], 'north'),
            (RIGHT_MID_X_23,  RIGHT_CORRIDOR_Y),
            (RIGHT_WALL_X[2], RIGHT_GAP_Y[2], 'north'),
            (RIGHT_PORT_X,    RIGHT_PORT_Y),
        ]


def build_return_route(side='left'):
    """Build return zigzag route from port back to start."""
    if side == 'left':
        return [
            (LEFT_WALL_X[2] + 0.20, LEFT_GAP_Y[2]),
            (LEFT_WALL_X[2], LEFT_GAP_Y[2], 'south'),
            (LEFT_MID_X_23, LEFT_GAP_Y[2]),
            (LEFT_MID_X_23, LEFT_GAP_Y[1]),
            (LEFT_WALL_X[1], LEFT_GAP_Y[1], 'south'),
            (LEFT_MID_X_12, LEFT_GAP_Y[1]),
            (LEFT_MID_X_12, LEFT_GAP_Y[0]),
            (LEFT_WALL_X[0], LEFT_GAP_Y[0], 'south'),
            (LEFT_START_X, LEFT_START_Y),
        ]
    else:
        return [
            (RIGHT_WALL_X[2] + 0.20, RIGHT_GAP_Y[2]),
            (RIGHT_WALL_X[2], RIGHT_GAP_Y[2], 'south'),
            (RIGHT_MID_X_23, RIGHT_GAP_Y[2]),
            (RIGHT_MID_X_23, RIGHT_GAP_Y[1]),
            (RIGHT_WALL_X[1], RIGHT_GAP_Y[1], 'south'),
            (RIGHT_MID_X_12, RIGHT_GAP_Y[1]),
            (RIGHT_MID_X_12, RIGHT_GAP_Y[0]),
            (RIGHT_WALL_X[0], RIGHT_GAP_Y[0], 'south'),
            (RIGHT_START_X, RIGHT_START_Y),
        ]


def load_waypoints_from_file(filepath):
    """Load waypoints from a JSON file produced by challenge_mapper.py.
    Returns (forward_route, return_route, port_data, coastal_side) or None."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        data = json.load(f)
    wp = data.get('waypoints', {})
    side = data.get('coastal_side', 'left')

    fwd = wp.get('forward_route')
    if fwd:
        fwd = [(m['x'], m['y']) for m in fwd]

    ret = wp.get('return_route')
    if ret:
        ret = [(m['x'], m['y']) for m in ret]

    port_data = {
        'cargo': wp.get('cargo'),
        'drop_off': wp.get('drop_off'),
        'maneuver_waypoints': wp.get('maneuver_waypoints', []),
        'pickup_approach': wp.get('pickup_approach'),
        'port_corners': wp.get('port_corners', []),
    }

    return fwd, ret, port_data, side


# ─────────────────────────────────────────────────────────────────
# Navigator Node
# ─────────────────────────────────────────────────────────────────
class ChallengeNavigator(Node):
    def __init__(self):
        super().__init__('challenge_navigator')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)
        self.status_pub = self.create_publisher(String, '/cv/nav_status', 10)

        # Standard subscribers
        self.last_odom = None
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        self.last_scan = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.last_imu = None
        self.create_subscription(Imu, '/bno055/imu', self._imu_cb, 10)

        # CV subscribers
        self.last_detections = []
        self.last_det_time = 0.0
        self.create_subscription(String, '/cv/detections', self._det_cb, 10)

        self.wall_ahead = False
        self.create_subscription(Bool, '/cv/wall_ahead', self._wall_cb, 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Calibration offsets
        self.yaw_offset = 0.0
        self.imu_yaw_offset = 0.0
        self.x_offset = 0.0
        self.y_offset = 0.0

    # ─── Callbacks ──────────────────────────────────────────────
    def _odom_cb(self, msg):
        self.last_odom = msg

    def _scan_cb(self, msg):
        self.last_scan = msg

    def _imu_cb(self, msg):
        self.last_imu = msg

    def _det_cb(self, msg):
        try:
            self.last_detections = json.loads(msg.data)
            self.last_det_time = time.time()
        except json.JSONDecodeError:
            self.last_detections = []

    def _wall_cb(self, msg):
        self.wall_ahead = msg.data

    def set_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(f'STATUS: {text}')

    # ─── CV helpers ─────────────────────────────────────────────
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

    def _detection_offset_x(self, det):
        cx = det['center'][0]
        return cx - CV_CENTER_X

    def _spin_cv(self, duration=0.15):
        t0 = time.time()
        while time.time() - t0 < duration:
            rclpy.spin_once(self, timeout_sec=0.05)

    def _validate_lifeboat_detection(self, det):
        """Filter out false-positive lifeboat detections.
        Returns True if the detection looks like a real lifeboat."""
        if det is None:
            return False
        area = det.get('area', 0)
        # Too large = probably cargo misidentified
        if area > LIFEBOAT_MAX_AREA:
            self.get_logger().info(
                f'LIFEBOAT FILTER: Rejected — area {area:.0f} > {LIFEBOAT_MAX_AREA} (too large, likely cargo)')
            return False
        # Too small = noise
        if area < LIFEBOAT_MIN_AREA:
            self.get_logger().info(
                f'LIFEBOAT FILTER: Rejected — area {area:.0f} < {LIFEBOAT_MIN_AREA} (too small)')
            return False
        # If we're near the port, cargo gets misidentified
        pose = self._get_pose()
        if pose is not None and pose[0] > LIFEBOAT_MAX_X:
            self.get_logger().info(
                f'LIFEBOAT FILTER: Rejected — robot at x={pose[0]:.2f} > {LIFEBOAT_MAX_X} (too close to port)')
            return False
        return True

    # ─── Pose helpers ───────────────────────────────────────────
    def _get_odom_yaw(self):
        if self.last_odom is None:
            return None
        q = self.last_odom.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _get_odom_xy(self):
        if self.last_odom is None:
            return None
        p = self.last_odom.pose.pose.position
        return (p.x, p.y)

    def _get_imu_yaw(self):
        if self.last_imu is None:
            return None
        q = self.last_imu.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _get_map_yaw(self):
        imu_yaw = self._get_imu_yaw()
        if imu_yaw is not None:
            return self._normalize_angle(imu_yaw + self.imu_yaw_offset)
        odom_yaw = self._get_odom_yaw()
        if odom_yaw is None:
            return None
        return self._normalize_angle(odom_yaw + self.yaw_offset)

    def _get_map_xy(self):
        odom_xy = self._get_odom_xy()
        if odom_xy is None:
            return None
        ox, oy = odom_xy
        if not hasattr(self, '_prev_odom_xy') or self._prev_odom_xy is None:
            c = math.cos(self.yaw_offset)
            s = math.sin(self.yaw_offset)
            self._map_x = ox * c - oy * s + self.x_offset
            self._map_y = ox * s + oy * c + self.y_offset
            self._prev_odom_xy = (ox, oy)
            return (self._map_x, self._map_y)
        dox = ox - self._prev_odom_xy[0]
        doy = oy - self._prev_odom_xy[1]
        self._prev_odom_xy = (ox, oy)
        imu_yaw = self._get_imu_yaw()
        odom_yaw = self._get_odom_yaw()
        if imu_yaw is not None and odom_yaw is not None:
            cur_yaw_off = self._normalize_angle(
                imu_yaw + self.imu_yaw_offset - odom_yaw)
        else:
            cur_yaw_off = self.yaw_offset
        c = math.cos(cur_yaw_off)
        s = math.sin(cur_yaw_off)
        self._map_x += dox * c - doy * s
        self._map_y += dox * s + doy * c
        return (self._map_x, self._map_y)

    def _get_pose(self):
        map_xy = self._get_map_xy()
        map_yaw = self._get_map_yaw()
        if map_xy is not None and map_yaw is not None:
            return (map_xy[0], map_xy[1], map_yaw)
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.3))
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)
            return (x, y, yaw)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

    def _spin_get_pose(self):
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)
            pose = self._get_pose()
            if pose is not None:
                return pose
            time.sleep(0.2)
        return None

    def _get_raw_tf_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.2))
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)
            return (x, y, yaw)
        except Exception:
            return None

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

    def _get_side_mins(self):
        return (self._lidar_sector_min(40, 120),
                self._lidar_sector_min(240, 320))

    def detect_coastal_side(self):
        """Detect left vs right coastal from lidar scan at startup.
        Returns 'left' or 'right'."""
        scan = self.last_scan
        if scan is None:
            return None
        n = len(scan.ranges)
        if n == 0:
            return None
        left_min = self._lidar_sector_min(60, 120)
        right_min = self._lidar_sector_min(240, 300)
        self.get_logger().info(
            f'Coastal detection: left={left_min:.2f}m, right={right_min:.2f}m')
        if left_min < right_min:
            return 'left'
        return 'right'

    # ─── Utilities ──────────────────────────────────────────────
    def _stop(self):
        self.cmd_pub.publish(Twist())

    @staticmethod
    def _normalize_angle(a):
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    @staticmethod
    def _yaw_to_quat(yaw):
        return (math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    # ─── Setup / calibration ────────────────────────────────────
    def calibrate_yaw_offset(self):
        odom_yaw = self._get_odom_yaw()
        odom_xy = self._get_odom_xy()
        if odom_yaw is None or odom_xy is None:
            self.get_logger().warn('No odom data for calibration')
            return
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.5))
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            tf_yaw = math.atan2(siny, cosy)
            tf_x = t.transform.translation.x
            tf_y = t.transform.translation.y
        except Exception:
            self.get_logger().warn('No TF for calibration')
            return
        self.yaw_offset = self._normalize_angle(tf_yaw - odom_yaw)
        ox, oy = odom_xy
        c = math.cos(self.yaw_offset)
        s = math.sin(self.yaw_offset)
        self.x_offset = tf_x - (ox * c - oy * s)
        self.y_offset = tf_y - (ox * s + oy * c)
        self._prev_odom_xy = None
        imu_yaw = self._get_imu_yaw()
        if imu_yaw is not None:
            self.imu_yaw_offset = self._normalize_angle(tf_yaw - imu_yaw)
            self.get_logger().info(
                f'Calibrated: imu_yaw={math.degrees(imu_yaw):.1f} '
                f'+ offset={math.degrees(self.imu_yaw_offset):.1f} '
                f'= map_yaw={math.degrees(tf_yaw):.1f}  '
                f'xy_offset=({self.x_offset:.3f},{self.y_offset:.3f})')
        else:
            self.get_logger().info(
                f'Calibrated: odom_yaw={math.degrees(odom_yaw):.1f} '
                f'+ offset={math.degrees(self.yaw_offset):.1f} '
                f'= map_yaw={math.degrees(tf_yaw):.1f}  '
                f'xy_offset=({self.x_offset:.3f},{self.y_offset:.3f}) [no IMU]')

    def wait_for_tf(self, timeout=30.0):
        self.get_logger().info('Waiting for map->base_footprint TF...')
        t0 = time.time()
        pose = None
        while pose is None:
            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=0.1)
            pose = self._get_pose()
            if time.time() - t0 > timeout:
                self.get_logger().error('TF map->base_footprint not available!')
                raise SystemExit(1)
        self.get_logger().info(
            f'TF OK: ({pose[0]:.3f}, {pose[1]:.3f}) yaw={math.degrees(pose[2]):.1f}')

    def verify_amcl_converged(self, expected_x, expected_y, expected_yaw,
                              tol_xy=0.15, tol_yaw=0.3):
        self.get_logger().info('Verifying AMCL convergence...')
        for attempt in range(60):
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.5)
            rclpy.spin_once(self, timeout_sec=0.1)
            pose = self._get_raw_tf_pose()
            if pose is None:
                continue
            x, y, yaw = pose
            dx = abs(x - expected_x)
            dy = abs(y - expected_y)
            dyaw = abs(self._normalize_angle(yaw - expected_yaw))
            self.get_logger().info(
                f'  AMCL check #{attempt+1}: pos=({x:.3f},{y:.3f}) '
                f'yaw={math.degrees(yaw):.1f} '
                f'err=({dx:.3f},{dy:.3f},{math.degrees(dyaw):.1f})')
            if dx < tol_xy and dy < tol_xy and dyaw < tol_yaw:
                self.get_logger().info('  AMCL converged!')
                return True
            if attempt % 15 == 14:
                self.get_logger().info('  Re-publishing initial pose...')
                self.publish_initial_pose(expected_x, expected_y, expected_yaw)
        self.get_logger().error('AMCL failed to converge after 60 attempts!')
        raise SystemExit(1)

    def publish_initial_pose(self, x, y, yaw):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        qz, qw = self._yaw_to_quat(yaw)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance[0]  = 0.01
        msg.pose.covariance[7]  = 0.01
        msg.pose.covariance[35] = 0.005
        for _ in range(5):
            self.initial_pose_pub.publish(msg)
            time.sleep(0.1)
        self.get_logger().info(
            f'Published initial pose: ({x:.3f}, {y:.3f}) yaw={math.degrees(yaw):.1f}')

    # ═══════════════════════════════════════════════════════════
    # CORE MOVEMENT PRIMITIVES
    # ═══════════════════════════════════════════════════════════

    def rotate_to(self, target_yaw, speed=None):
        if speed is None:
            speed = ROTATE_SPEED
        cmd = Twist()
        step = 0
        while rclpy.ok():
            pose = self._spin_get_pose()
            if pose is None:
                time.sleep(DT)
                continue
            _, _, yaw = pose
            err = self._normalize_angle(target_yaw - yaw)
            if abs(err) < YAW_TOL:
                break
            spd = min(speed, max(0.06, abs(err) * 0.8))
            cmd.angular.z = spd if err > 0 else -spd
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            step += 1
            if step % 15 == 0:
                self.get_logger().info(
                    f'    ROT: yaw={math.degrees(yaw):.1f} '
                    f'target={math.degrees(target_yaw):.1f} '
                    f'err={math.degrees(err):.1f}')
            time.sleep(DT)
        self._stop()
        time.sleep(0.1)

    def drive_to(self, tx, ty, gate_dir=None):
        self._arrival_clean = False
        cmd = Twist()
        step = 0
        min_dist = float('inf')
        initial_dist = None
        last_progress_time = time.time()
        best_progress = float('-inf')

        while rclpy.ok():
            pose = self._spin_get_pose()
            if pose is None:
                time.sleep(DT)
                continue
            x, y, yaw = pose
            dx = tx - x
            dy = ty - y
            dist = math.hypot(dx, dy)

            if gate_dir is not None:
                crossed = (x > tx) if gate_dir == 'north' else (x < tx)
                if crossed:
                    self.get_logger().info(
                        f'    GATE: crossed x={tx:.3f} at pos=({x:.3f},{y:.3f})')
                    self._arrival_clean = True
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                    self.cmd_pub.publish(cmd)
                    time.sleep(0.3)
                    break

            if gate_dir is None and dist < DIST_TOL:
                self._arrival_clean = True
                break

            if gate_dir is None:
                if initial_dist is None:
                    initial_dist = dist
                if dist < min_dist:
                    min_dist = dist
                if min_dist < initial_dist * 0.75 and dist > min_dist + 0.04:
                    self.get_logger().info('    DRV: overshoot — moving on')
                    break

            now = time.time()
            if gate_dir == 'north':
                progress = x
            elif gate_dir == 'south':
                progress = -x
            else:
                progress = -dist
            if progress > best_progress + 0.01:
                best_progress = progress
                last_progress_time = now
            if now - last_progress_time > STUCK_TIME:
                self.get_logger().warn(
                    f'    DRV: stuck for {STUCK_TIME:.1f}s — moving on')
                break

            desired_yaw = math.atan2(dy, dx)
            yaw_err = self._normalize_angle(desired_yaw - yaw)

            step += 1
            if step % 20 == 0:
                f_r = self._get_front_min()
                l_r, r_r = self._get_side_mins()
                self.get_logger().info(
                    f'    DRV: pos=({x:.3f},{y:.3f}) dist={dist:.3f} '
                    f'yaw_err={math.degrees(yaw_err):.1f} '
                    f'lidar F={f_r:.2f} L={l_r:.2f} R={r_r:.2f}')

            speed = LINEAR_SPEED
            if dist < 0.30:
                speed = max(0.04, LINEAR_SPEED * (dist / 0.30))
            if gate_dir is not None:
                gate_dist = abs(tx - x)
                if gate_dist < 0.25:
                    speed = min(speed, max(0.04, LINEAR_SPEED * (gate_dist / 0.25)))
            if abs(yaw_err) > 0.5:
                speed *= 0.5

            front_min = self._get_front_min()
            if front_min < LIDAR_FRONT_STOP:
                speed = 0.0
                self.get_logger().warn(f'    EMERGENCY STOP: front={front_min:.2f}m')
            elif front_min < LIDAR_SLOW_DIST:
                speed *= max(0.3, front_min / LIDAR_SLOW_DIST)

            if abs(yaw_err) < STEER_DEADBAND:
                steer = 0.0
            else:
                steer = max(-MAX_STEER, min(MAX_STEER, yaw_err * STEER_GAIN))

            left_min, right_min = self._get_side_mins()
            side_corr = 0.0
            if left_min < LIDAR_SIDE_WARN:
                side_corr -= (LIDAR_SIDE_WARN - left_min) * LIDAR_SIDE_GAIN
            if right_min < LIDAR_SIDE_WARN:
                side_corr += (LIDAR_SIDE_WARN - right_min) * LIDAR_SIDE_GAIN
            if side_corr != 0.0:
                steer = max(-MAX_STEER, min(MAX_STEER, steer + side_corr))

            cmd.linear.x = speed
            cmd.angular.z = steer
            self.cmd_pub.publish(cmd)

            for _ in range(3):
                rclpy.spin_once(self, timeout_sec=0.001)
            time.sleep(DT)
        self._stop()

    def goto(self, tx, ty, label='', gate_dir=None, recalibrate=True):
        t0 = time.time()
        pose = self._spin_get_pose()
        if pose is None:
            self.get_logger().warn('No TF pose – skipping waypoint')
            return
        x, y, _ = pose

        if gate_dir is not None:
            already_through = (x > tx) if gate_dir == 'north' else (x < tx)
            if already_through:
                self.get_logger().info(
                    f'{label}  Already past gate x={tx:.3f} (cur x={x:.3f}) — skipping')
                return

        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        if gate_dir is None and dist < DIST_TOL:
            if recalibrate:
                self._recalibrate_yaw_offset()
            return
        target_yaw = math.atan2(dy, dx)

        kind = 'GATE' if gate_dir else 'DEST'
        self.get_logger().info(
            f'{label}  [{kind}] Rotate -> {math.degrees(target_yaw):.0f} '
            f'then drive {dist:.2f} m to ({tx:.3f}, {ty:.3f})')

        self.rotate_to(target_yaw)
        time.sleep(0.15)

        if time.time() - t0 > GOTO_TIMEOUT:
            self.get_logger().warn(f'{label}  Timeout after rotation — moving on')
            self._stop()
            return

        self.drive_to(tx, ty, gate_dir=gate_dir)
        time.sleep(0.05)

        if gate_dir is None and recalibrate and getattr(self, '_arrival_clean', False):
            self._recalibrate_yaw_offset()

    def _recalibrate_yaw_offset(self):
        odom_yaw = self._get_odom_yaw()
        odom_xy = self._get_odom_xy()
        if odom_yaw is None or odom_xy is None:
            return
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.2))
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            tf_yaw = math.atan2(siny, cosy)
            tf_x = t.transform.translation.x
            tf_y = t.transform.translation.y
        except Exception:
            return
        old_offset = self.yaw_offset
        new_offset = self._normalize_angle(tf_yaw - odom_yaw)
        delta = abs(self._normalize_angle(new_offset - old_offset))
        if delta > math.radians(8):
            self.get_logger().warn(
                f'Recalibration rejected: yaw change {math.degrees(delta):.1f} > 8')
            return
        cur_xy = self._get_map_xy()
        if cur_xy is not None:
            xy_jump = math.hypot(tf_x - cur_xy[0], tf_y - cur_xy[1])
            if xy_jump > 0.3:
                self.get_logger().warn(
                    f'Recalibration rejected: XY jump {xy_jump:.3f}m > 0.3m')
                return
        self.yaw_offset = new_offset
        ox, oy = odom_xy
        c = math.cos(self.yaw_offset)
        s = math.sin(self.yaw_offset)
        self.x_offset = tf_x - (ox * c - oy * s)
        self.y_offset = tf_y - (ox * s + oy * c)
        self._prev_odom_xy = None
        self.get_logger().info(
            f'Recalibrated yaw offset: {math.degrees(old_offset):.1f} -> '
            f'{math.degrees(self.yaw_offset):.1f}')

    def follow_route(self, waypoints, label='', recalibrate=True):
        total = len(waypoints)
        for i, wp in enumerate(waypoints):
            self.set_status(f'{label} waypoint {i+1}/{total}')
            if len(wp) == 3:
                wx, wy, gate_dir = wp
            else:
                wx, wy = wp
                gate_dir = None
            self.goto(wx, wy, label=f'[{label} {i+1}/{total}]',
                      gate_dir=gate_dir, recalibrate=recalibrate)
        self.get_logger().info(f'{label} route complete!')

    # ═══════════════════════════════════════════════════════════
    # CV-GUIDED BEHAVIOURS
    # ═══════════════════════════════════════════════════════════

    def cv_scan_for(self, label, timeout=None):
        """Slowly rotate scanning for a detection. Returns det or None."""
        if timeout is None:
            timeout = CV_SCAN_TIMEOUT
        self.get_logger().info(f'CV SCAN: Looking for {label}...')
        t0 = time.time()
        cmd = Twist()
        total_rotated = 0.0
        last_yaw = None

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin_cv(0.1)
            det = self._get_best_detection(label)
            if det is not None:
                self.get_logger().info(
                    f'CV SCAN: Found {label}! conf={det["confidence"]:.2f} '
                    f'area={det["area"]:.0f}')
                self._stop()
                return det

            cmd.angular.z = CV_ROTATE
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)

            pose = self._get_pose()
            if pose is not None:
                cur_yaw = pose[2]
                if last_yaw is not None:
                    d_yaw = abs(self._normalize_angle(cur_yaw - last_yaw))
                    total_rotated += d_yaw
                last_yaw = cur_yaw
                if total_rotated > 2 * math.pi + 0.5:
                    self.get_logger().warn(f'CV SCAN: Full rotation, {label} not found')
                    break
            time.sleep(DT)

        self._stop()
        self.get_logger().warn(f'CV SCAN: Timeout looking for {label}')
        return None

    def cv_center_on(self, label, timeout=8.0):
        """Rotate to center the detected object in camera frame."""
        self.get_logger().info(f'CV CENTER: Centering on {label}...')
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

            offset = self._detection_offset_x(det)
            if abs(offset) < CV_CENTER_TOL:
                self._stop()
                self.get_logger().info(
                    f'CV CENTER: {label} centered (offset={offset:.0f}px)')
                return det

            rot_speed = min(CV_ROTATE, max(0.05, abs(offset) / CAM_WIDTH * 0.6))
            cmd.angular.z = -rot_speed if offset > 0 else rot_speed
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            time.sleep(DT)

        self._stop()
        self.get_logger().warn(f'CV CENTER: Timeout centering {label}')
        return None

    def cv_approach(self, label, target_area, timeout=None):
        """Drive toward object keeping it centered until area >= target_area."""
        if timeout is None:
            timeout = CV_APPROACH_TIMEOUT
        self.get_logger().info(
            f'CV APPROACH: Driving toward {label} until area>{target_area:.0f}...')
        t0 = time.time()
        cmd = Twist()
        lost_count = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin_cv(0.05)

            front_min = self._get_front_min()
            if front_min < LIDAR_FRONT_STOP:
                self._stop()
                self.get_logger().warn(
                    f'CV APPROACH: WALL at {front_min:.2f}m — backing up!')
                self._backup_from_wall(0.15)
                return self._get_best_detection(label)

            det = self._get_best_detection(label)
            if det is None:
                lost_count += 1
                if lost_count > 30:
                    self._stop()
                    self.get_logger().warn(f'CV APPROACH: Lost {label}')
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
                    f'CV APPROACH: {label} reached! area={det["area"]:.0f}')
                return det

            offset = self._detection_offset_x(det)
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

            if int((time.time() - t0) * 10) % 20 == 0:
                self.get_logger().info(
                    f'    CV: area={det["area"]:.0f} offset={offset:.0f} '
                    f'front={front_min:.2f}')
            time.sleep(DT)

        self._stop()
        self.get_logger().warn(f'CV APPROACH: Timeout approaching {label}')
        return None

    def _backup_from_wall(self, distance=0.20):
        """Emergency: reverse away from a wall."""
        self.get_logger().warn(f'BACKUP: Reversing {distance:.2f}m from wall...')
        cmd = Twist()
        cmd.linear.x = -LINEAR_SPEED * 0.6
        t0 = time.time()
        # Rough time estimate: distance / speed
        duration = distance / (LINEAR_SPEED * 0.6)
        while time.time() - t0 < duration:
            self.cmd_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(DT)
        self._stop()
        time.sleep(0.2)

    def cv_approach_to_frame_bottom(self, label, timeout=15.0):
        """Drive toward object until its bounding box bottom reaches CV_BOTTOM_Y.
        Has aggressive lidar safety — backs up if it hits a wall."""
        self.get_logger().info(
            f'CV BOTTOM: Approaching {label} until at frame bottom...')
        t0 = time.time()
        cmd = Twist()
        lost_count = 0

        while rclpy.ok() and time.time() - t0 < timeout:
            self._spin_cv(0.05)

            front_min = self._get_front_min()
            left_min, right_min = self._get_side_mins()
            side_min = min(left_min, right_min)

            # Emergency stop + backup if too close to any wall
            if front_min < LIDAR_FRONT_STOP:
                self._stop()
                self.get_logger().warn(
                    f'CV BOTTOM: WALL at {front_min:.2f}m — backing up!')
                self._backup_from_wall(0.20)
                return None  # Abort this approach — likely a false target

            if side_min < LIDAR_FRONT_STOP:
                self._stop()
                self.get_logger().warn(
                    f'CV BOTTOM: Side wall at {side_min:.2f}m — stopping')
                return None

            det = self._get_best_detection(label)
            if det is None:
                lost_count += 1
                if lost_count > 30:
                    self._stop()
                    self.get_logger().warn(f'CV BOTTOM: Lost {label}')
                    return None
                cmd.linear.x = CV_LINEAR * 0.3
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                time.sleep(DT)
                continue
            lost_count = 0

            bbox = det.get('bbox', [0, 0, 0, 0])
            bottom_edge = bbox[1] + bbox[3]
            if bottom_edge >= CV_BOTTOM_Y:
                self._stop()
                self.get_logger().info(
                    f'CV BOTTOM: {label} at frame bottom! '
                    f'bbox_bottom={bottom_edge:.0f}')
                return det

            offset = self._detection_offset_x(det)
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

            if int((time.time() - t0) * 10) % 20 == 0:
                self.get_logger().info(
                    f'    CV BOTTOM: bbox_bottom={bottom_edge:.0f} '
                    f'offset={offset:.0f} front={front_min:.2f}')
            time.sleep(DT)

        self._stop()
        self.get_logger().warn(f'CV BOTTOM: Timeout approaching {label}')
        return None

    def cv_identify_cargo_orientation(self):
        """Scan for cargo, center on it, determine orientation.
        Returns (detection, 'sideways'|'endwise') or (None, None)."""
        det = self.cv_scan_for('cargo')
        if det is None:
            return None, None
        det = self.cv_center_on('cargo')
        if det is None:
            return None, None
        aspect = det.get('aspect_ratio', 1.0)
        orientation = 'sideways' if aspect > 1.2 else 'endwise'
        self.get_logger().info(
            f'CARGO ORIENTATION: aspect_ratio={aspect:.2f} -> {orientation}')
        return det, orientation

    def cv_find_and_approach(self, label, target_area):
        """Full CV pipeline: scan -> center -> approach."""
        det = self.cv_scan_for(label)
        if det is None:
            return None
        det = self.cv_center_on(label)
        if det is None:
            return None
        return self.cv_approach(label, target_area)

    # ═══════════════════════════════════════════════════════════
    # LIFEBOAT INTERACTION (new: 10s pause, no 360° spin)
    # ═══════════════════════════════════════════════════════════

    def handle_lifeboat(self):
        """Center on lifeboat, approach until at frame bottom,
        then pause for LIFEBOAT_PAUSE_SECONDS to simulate rescue.
        Returns True if interaction completed, False if aborted."""
        # Safety check: are we already too close to a wall?
        front_min = self._get_front_min()
        if front_min < LIDAR_FRONT_STOP * 2:
            self.get_logger().warn(
                f'LIFEBOAT: Wall too close ({front_min:.2f}m) — aborting approach')
            return False

        centered = self.cv_center_on('lifeboat')
        if centered is None:
            self.get_logger().warn('LIFEBOAT: Lost during centering')
            return False

        # Re-validate after centering — might have been a false positive
        det = self._get_best_detection('lifeboat')
        if det is not None and not self._validate_lifeboat_detection(det):
            self.get_logger().warn('LIFEBOAT: Post-center validation failed — aborting')
            return False

        result = self.cv_approach_to_frame_bottom('lifeboat')
        if result is None:
            self.get_logger().warn('LIFEBOAT: Approach failed/wall hit — aborting')
            return False

        self.get_logger().info(
            f'LIFEBOAT: At frame bottom — pausing {LIFEBOAT_PAUSE_SECONDS:.0f}s '
            f'for rescue simulation...')
        self.set_status(f'LIFEBOAT RESCUE — pausing {LIFEBOAT_PAUSE_SECONDS:.0f}s')
        self._stop()

        # Pause while keeping ROS alive
        t0 = time.time()
        while time.time() - t0 < LIFEBOAT_PAUSE_SECONDS:
            rclpy.spin_once(self, timeout_sec=0.2)
            remaining = LIFEBOAT_PAUSE_SECONDS - (time.time() - t0)
            if int(remaining) % 3 == 0 and abs(remaining - int(remaining)) < 0.3:
                self.get_logger().info(f'  Rescue in progress... {remaining:.0f}s remaining')

        self.get_logger().info('LIFEBOAT: Rescue simulation complete!')
        self.set_status('LIFEBOAT RESCUE COMPLETE')
        return True

    # ═══════════════════════════════════════════════════════════
    # ROUTE WITH LIFEBOAT SCANNING (both directions)
    # ═══════════════════════════════════════════════════════════

    def follow_route_with_lifeboat_scan(self, waypoints, label='', recalibrate=True):
        """Follow a route, checking for lifeboat between every waypoint.
        No lifeboat_handled flag — triggers EVERY time it's spotted."""
        total = len(waypoints)

        for i, wp in enumerate(waypoints):
            self.set_status(f'{label} waypoint {i+1}/{total}')
            if len(wp) == 3:
                wx, wy, gate_dir = wp
            else:
                wx, wy = wp
                gate_dir = None

            # Check for lifeboat before driving to next waypoint
            self._spin_cv(0.3)
            det = self._get_best_detection('lifeboat')
            if det is not None and self._validate_lifeboat_detection(det):
                self.set_status(f'{label} LIFEBOAT spotted!')
                self.get_logger().info(
                    f'LIFEBOAT DETECTED on {label} segment {i+1}! '
                    f'area={det["area"]:.0f} conf={det["confidence"]:.2f}')
                success = self.handle_lifeboat()
                if not success:
                    self.get_logger().warn('LIFEBOAT: Approach failed/aborted — resuming route')
                self.set_status(f'{label} waypoint {i+1}/{total} (resuming)')

            self.goto(wx, wy, label=f'[{label} {i+1}/{total}]',
                      gate_dir=gate_dir, recalibrate=recalibrate)

        self.get_logger().info(f'{label} route complete!')

    # ═══════════════════════════════════════════════════════════
    # PORT INTERACTION (drop cargo → maneuver → pickup target)
    # ═══════════════════════════════════════════════════════════

    def port_interaction(self, port_data=None):
        """Full port interaction sequence:
        1. Identify cargo orientation
        2. Approach from long-axis side (wrong pickup angle = drop-off angle)
        3. Drop starting cargo (pause)
        4. Back up
        5. Maneuver around dropped cargo
        6. Approach target cargo from correct pickup side
        7. Pick up target cargo (pause)
        """
        self.set_status('PORT: Identifying cargo...')

        # Step 1: Find and identify target cargo
        cargo_det, cargo_orient = self.cv_identify_cargo_orientation()
        if cargo_det is not None:
            self.get_logger().info(
                f'PORT: Cargo found — {cargo_orient}, '
                f'area={cargo_det["area"]:.0f}')
        else:
            self.get_logger().warn('PORT: Cargo not detected — blind approach')
            cargo_orient = 'unknown'

        # Step 2: Approach cargo from the long-axis side (drop-off angle)
        # This is intentionally the "wrong" angle for pickup
        self.set_status(f'PORT: Approaching cargo ({cargo_orient}) for drop-off')

        if cargo_det is not None:
            # CV approach toward cargo
            result = self.cv_approach('cargo', CV_CARGO_AREA_CLOSE, timeout=15.0)
            if result is not None:
                self.get_logger().info(f'PORT: Near cargo for drop-off')
        else:
            # Blind: just drive forward into port area
            pose = self._spin_get_pose()
            if pose:
                self.goto(pose[0] + 0.3, pose[1], label='[PORT blind fwd]',
                          recalibrate=False)

        # Step 3: Simulate cargo drop
        self.set_status('PORT: DROPPING CARGO')
        self.get_logger().info(
            f'PORT: Dropping cargo — pausing {CARGO_DROP_PAUSE:.0f}s...')
        self._stop()
        t0 = time.time()
        while time.time() - t0 < CARGO_DROP_PAUSE:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info('PORT: CARGO DROPPED')

        # Step 4: Back up from drop position
        self.set_status('PORT: Backing up from drop')
        self.get_logger().info('PORT: Backing up...')
        cmd = Twist()
        cmd.linear.x = -LINEAR_SPEED * 0.7
        t0 = time.time()
        while time.time() - t0 < 2.5:
            self.cmd_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(DT)
        self._stop()
        time.sleep(0.3)

        # Step 5: Maneuver around dropped cargo
        # If we have mapper waypoints, use those
        if port_data and port_data.get('maneuver_waypoints'):
            self.set_status('PORT: Maneuvering around (waypoints)')
            for j, mwp in enumerate(port_data['maneuver_waypoints']):
                self.goto(mwp['x'], mwp['y'],
                          label=f'[PORT maneuver {j+1}]', recalibrate=False)
        else:
            # Default maneuver: rotate ~90° to side, drive parallel, rotate back
            self.set_status('PORT: Maneuvering around (default)')
            pose = self._spin_get_pose()
            if pose:
                x, y, yaw = pose
                # Move sideways: perpendicular to our current heading
                # Determine which side to go (away from arena wall)
                side_offset = 0.35  # distance to move sideways
                perp_yaw = yaw + math.pi / 2  # left
                side_x = x + side_offset * math.cos(perp_yaw)
                side_y = y + side_offset * math.sin(perp_yaw)
                self.goto(side_x, side_y, label='[PORT go-around 1]',
                          recalibrate=False)
                # Drive forward past the dropped cargo
                fwd_dist = 0.5
                fwd_x = side_x + fwd_dist * math.cos(yaw)
                fwd_y = side_y + fwd_dist * math.sin(yaw)
                self.goto(fwd_x, fwd_y, label='[PORT go-around 2]',
                          recalibrate=False)
                # Come back inline toward cargo
                inline_x = fwd_x - side_offset * math.cos(perp_yaw)
                inline_y = fwd_y - side_offset * math.sin(perp_yaw)
                self.goto(inline_x, inline_y, label='[PORT go-around 3]',
                          recalibrate=False)

        # Step 6: Now face and approach target cargo from the correct pickup side
        self.set_status('PORT: CV approach for pickup')

        if port_data and port_data.get('pickup_approach'):
            pa = port_data['pickup_approach']
            self.goto(pa['x'], pa['y'], label='[PORT pickup pos]',
                      recalibrate=False)
            if 'yaw' in pa:
                self.rotate_to(pa['yaw'])

        # Use CV to find and approach the target cargo
        cargo_result = self.cv_find_and_approach('cargo', CV_CARGO_AREA_CLOSE)
        if cargo_result is not None:
            self.get_logger().info(
                f'PORT: Target cargo reached! area={cargo_result["area"]:.0f}')
            # Nudge forward to ensure contact
            cmd = Twist()
            cmd.linear.x = 0.06
            self.cmd_pub.publish(cmd)
            time.sleep(1.5)
            self._stop()
        else:
            self.get_logger().warn('PORT: CV pickup failed — blind nudge')
            cmd = Twist()
            cmd.linear.x = LINEAR_SPEED * 0.5
            self.cmd_pub.publish(cmd)
            time.sleep(2.0)
            self._stop()

        # Step 7: Simulate pickup
        self.set_status('PORT: PICKING UP CARGO')
        self.get_logger().info(
            f'PORT: Picking up cargo — pausing {CARGO_PICKUP_PAUSE:.0f}s...')
        t0 = time.time()
        while time.time() - t0 < CARGO_PICKUP_PAUSE:
            rclpy.spin_once(self, timeout_sec=0.2)
        self.get_logger().info('PORT: CARGO PICKED UP')
        self.set_status('PORT: Cargo secured')


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    nav = ChallengeNavigator()

    # Parse optional --waypoints argument
    waypoints_file = None
    args = sys.argv[1:]
    if '--waypoints' in args:
        idx = args.index('--waypoints')
        if idx + 1 < len(args):
            waypoints_file = args[idx + 1]

    nav.get_logger().info('=== Challenge Navigator — Optimized ===')

    # ═══════════════════════════════════════════════════════════════
    # PHASE A: DETECT — Auto-detect coastal side
    # ═══════════════════════════════════════════════════════════════
    nav.set_status('Phase A: Detecting coastal side...')

    # Wait for lidar data
    for _ in range(30):
        rclpy.spin_once(nav, timeout_sec=0.2)
        if nav.last_scan is not None:
            break

    coastal_side = nav.detect_coastal_side()
    if coastal_side is None:
        nav.get_logger().warn('Could not detect coastal side — defaulting to left')
        coastal_side = 'left'
    nav.get_logger().info(f'=== Coastal side: {coastal_side.upper()} ===')
    nav.set_status(f'Phase A: {coastal_side.upper()} coastal detected')

    # Load waypoints from file or use defaults
    port_data = None
    fwd_route = None
    ret_route = None

    if waypoints_file:
        result = load_waypoints_from_file(waypoints_file)
        if result:
            fwd_route, file_ret, port_data, file_side = result
            if fwd_route:
                nav.get_logger().info(
                    f'Loaded {len(fwd_route)} forward waypoints from file')
            if file_ret:
                ret_route = file_ret
                nav.get_logger().info(
                    f'Loaded {len(ret_route)} return waypoints from file')
            if file_side:
                coastal_side = file_side
                nav.get_logger().info(f'Using file coastal side: {coastal_side}')
    else:
        # Try default file paths
        for default_path in [
            os.path.expanduser(f'~/ros2_ws/src/eced3901/config/challenge_waypoints_{coastal_side}.json'),
            os.path.expanduser('~/ros2_ws/src/eced3901/config/challenge_waypoints_left.json'),
        ]:
            if os.path.exists(default_path):
                result = load_waypoints_from_file(default_path)
                if result:
                    fwd_route, file_ret, port_data, file_side = result
                    if file_ret:
                        ret_route = file_ret
                    nav.get_logger().info(f'Loaded waypoints from {default_path}')
                    break

    if fwd_route is None:
        fwd_route = build_forward_route(coastal_side)
    if ret_route is None:
        ret_route = build_return_route(coastal_side)

    # Set start position based on coastal side
    if coastal_side == 'left':
        start_x, start_y, start_yaw = LEFT_START_X, LEFT_START_Y, LEFT_START_YAW
    else:
        start_x, start_y, start_yaw = RIGHT_START_X, RIGHT_START_Y, RIGHT_START_YAW

    nav.get_logger().info(f'Start: ({start_x:.3f},{start_y:.3f}) yaw={start_yaw:.1f}')
    nav.get_logger().info(f'Forward route: {len(fwd_route)} waypoints')
    nav.get_logger().info(f'Return route: {len(ret_route)} waypoints')

    # Wait for CV pipeline
    nav.get_logger().info('Waiting for CV pipeline...')
    t0 = time.time()
    while time.time() - t0 < 10.0:
        rclpy.spin_once(nav, timeout_sec=0.2)
        if nav.last_det_time > 0:
            break
    if nav.last_det_time > 0:
        nav.get_logger().info('CV pipeline active!')
    else:
        nav.get_logger().warn('CV pipeline not detected — continuing without CV')

    # Publish initial pose for AMCL
    nav.publish_initial_pose(start_x, start_y, start_yaw)
    time.sleep(3.0)
    nav.wait_for_tf()
    nav.verify_amcl_converged(start_x, start_y, start_yaw)
    nav.calibrate_yaw_offset()

    # ═══════════════════════════════════════════════════════════════
    # PHASE B: FORWARD — Zigzag to port with lifeboat scanning
    # ═══════════════════════════════════════════════════════════════
    nav.set_status(f'Phase B: FORWARD to port ({len(fwd_route)} wp) — lifeboat scan ON')
    nav.follow_route_with_lifeboat_scan(fwd_route, label='FWD', recalibrate=False)
    nav.set_status('Phase B: Reached port!')

    # ═══════════════════════════════════════════════════════════════
    # PHASE C: PORT — Drop cargo, maneuver, pickup target
    # ═══════════════════════════════════════════════════════════════
    nav.set_status('Phase C: PORT INTERACTION')
    nav.port_interaction(port_data=port_data)
    nav.set_status('Phase C: Port interaction complete!')
    time.sleep(1.0)

    # ═══════════════════════════════════════════════════════════════
    # PHASE D: RETURN — Zigzag home with lifeboat scanning
    # ═══════════════════════════════════════════════════════════════
    nav.calibrate_yaw_offset()
    nav.set_status(f'Phase D: RETURN home ({len(ret_route)} wp) — lifeboat scan ON')
    nav.follow_route_with_lifeboat_scan(ret_route, label='RET', recalibrate=False)

    # ═══════════════════════════════════════════════════════════════
    # PHASE E: DONE
    # ═══════════════════════════════════════════════════════════════
    nav.set_status('COMPLETE')
    nav._stop()
    nav.get_logger().info('=== Challenge complete! ===')
    nav.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
