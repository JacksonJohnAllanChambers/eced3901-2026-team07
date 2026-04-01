#! /usr/bin/env python3
"""
ECED3901 Challenge Arena – Left Coastal Navigator (SLAM Real Map)
=================================================================
Navigates the left coastal zigzag using the SLAM-captured real map.

The SLAM map coordinate frame is rotated 90° from the arena frame:
  • SLAM +X = arena NORTH  (robot forward at start)
  • SLAM +Y = arena WEST   (robot left at start)
  • Robot starts at approx (0, 0) with yaw=0 in the SLAM map

Gap positions were measured from the real SLAM map:
  • Wall 1 (x≈0.85): gap at y=0.054  (width 0.58m)
  • Wall 2 (x≈1.80): gap at y=0.653  (width 0.53m)
  • Wall 3 (x≈2.70): gap at y=0.116  (width 0.60m)

Usage:
  1. Launch robot.launch.py (motor_ws)
  2. Launch arena_real_amcl.launch.py (uses arena_real_map)
  3. ros2 run eced3901 arena_realmap_nav.py
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
import tf2_ros

# ─────────────────────────────────────────────────────────────────────
# SLAM map coordinate constants
# ─────────────────────────────────────────────────────────────────────
# Directions in SLAM frame (robot starts facing +X = arena north)
NORTH = 0.0              # +X direction
SOUTH = math.pi          # -X direction
EAST  = -math.pi / 2     # -Y direction (arena east)
WEST  = math.pi / 2      # +Y direction (arena west)

# Starting pose in SLAM map frame
START_X   = 0.0
START_Y   = 0.0
START_YAW = 0.0   # facing +X = north

# Port location (left_coastal: arena 3ft,12.5ft → SLAM ~3.50, 0.0)
PORT_X = 3.40
PORT_Y = 0.0

# Zigzag wall X-positions in SLAM frame (crossing north-south)
WALL_X = (0.85, 1.80, 2.70)

# Gap Y-positions measured from SLAM map analysis
GAP_Y = (0.054, 0.653, 0.116)  # wall1, wall2, wall3

# Midpoints between walls (for safe intermediate waypoints)
MID_X_12 = (WALL_X[0] + WALL_X[1]) / 2   # 1.325
MID_X_23 = (WALL_X[1] + WALL_X[2]) / 2   # 2.25

# Corridor bounds (outer wall at y≈0.9, inner edge at y≈-0.25)
CORRIDOR_CENTER_Y = 0.35

# Speeds — conservative for real robot
LINEAR_SPEED  = 0.13     # m/s
ROTATE_SPEED  = 0.30     # rad/s

# Tolerances
YAW_TOL       = 0.05     # rad (~2.9°)
DIST_TOL      = 0.15     # m

# Driving behaviour
STEER_GAIN    = 1.2
MAX_STEER     = 0.5      # rad/s max angular correction
STEER_DEADBAND = 0.03    # rad
STUCK_TIME    = 5.0      # seconds without progress → skip
GOTO_TIMEOUT  = 25.0     # seconds max per goto() call

# Lidar safety
LIDAR_SIDE_WARN   = 0.13   # m
LIDAR_SIDE_GAIN   = 3.0
LIDAR_SLOW_DIST   = 0.55   # m
LIDAR_FRONT_STOP  = 0.12   # m

# Control loop
DT = 1.0 / 30.0


# ─────────────────────────────────────────────────────────────────────
# Route builders (SLAM coordinates)
# ─────────────────────────────────────────────────────────────────────
def build_forward_route():
    """Forward through left coastal zigzag: gap1 → gap2 → gap3 → port."""
    return [
        # Cross wall 1 at gap y=0.054
        (WALL_X[0], GAP_Y[0], 'north'),
        # Intermediate: clear wall 1, steer toward gap 2
        (MID_X_12,  CORRIDOR_CENTER_Y),
        # Cross wall 2 at gap y=0.653
        (WALL_X[1], GAP_Y[1], 'north'),
        # Intermediate: clear wall 2, steer toward gap 3
        (MID_X_23,  CORRIDOR_CENTER_Y),
        # Cross wall 3 at gap y=0.116
        (WALL_X[2], GAP_Y[2], 'north'),
        # Port delivery zone
        (PORT_X,    PORT_Y),
    ]


def build_return_route():
    """Return through left coastal zigzag: port → gap3 → gap2 → gap1 → start."""
    return [
        # Approach wall 3 from north side
        (WALL_X[2] + 0.20, GAP_Y[2]),
        # Cross wall 3 going south
        (WALL_X[2], GAP_Y[2], 'south'),
        # Intermediate between wall 3 and wall 2
        (MID_X_23, GAP_Y[2]),
        (MID_X_23, GAP_Y[1]),
        # Cross wall 2 going south
        (WALL_X[1], GAP_Y[1], 'south'),
        # Intermediate between wall 2 and wall 1
        (MID_X_12, GAP_Y[1]),
        (MID_X_12, GAP_Y[0]),
        # Cross wall 1 going south
        (WALL_X[0], GAP_Y[0], 'south'),
        # Home
        (START_X, START_Y),
    ]


# ─────────────────────────────────────────────────────────────────────
# Navigator node
# ─────────────────────────────────────────────────────────────────────
class DirectNavigator(Node):
    def __init__(self):
        super().__init__('realmap_navigator')

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        # Subscribers
        self.last_odom = None
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        self.last_scan = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.last_imu = None
        self.create_subscription(Imu, '/imu/imu', self._imu_cb, 10)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Yaw offset calibration
        self.yaw_offset = 0.0
        self.imu_yaw_offset = 0.0
        self.x_offset = 0.0
        self.y_offset = 0.0

    # ─── Callbacks ─────────────────────────────────────────────────
    def _odom_cb(self, msg):
        self.last_odom = msg

    def _scan_cb(self, msg):
        self.last_scan = msg

    def _imu_cb(self, msg):
        self.last_imu = msg

    # ─── Pose helpers ──────────────────────────────────────────────
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
        odom_yaw = self._get_odom_yaw()
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

    # ─── Lidar helpers ─────────────────────────────────────────────
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
        """Returns (left_min, right_min)."""
        return (self._lidar_sector_min(40, 120),
                self._lidar_sector_min(240, 320))

    # ─── Helpers ───────────────────────────────────────────────────
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

    # ─── Setup ─────────────────────────────────────────────────────
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

    # ─── Core movement primitives ──────────────────────────────────
    def rotate_to(self, target_yaw):
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
            speed = min(ROTATE_SPEED, max(0.06, abs(err) * 0.8))
            cmd.angular.z = speed if err > 0 else -speed
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            step += 1
            if step % 15 == 0:
                self.get_logger().info(
                    f'    ROT: yaw={math.degrees(yaw):.1f} '
                    f'target={math.degrees(target_yaw):.1f} '
                    f'err={math.degrees(err):.1f} wz={cmd.angular.z:+.2f}')
            time.sleep(DT)
        self._stop()
        time.sleep(0.1)

    def drive_to(self, tx, ty, gate_dir=None):
        """Drive toward (tx, ty). If gate_dir is set, the target is a zigzag
        wall at X=tx with the gap at Y=ty. 'north' means crossing +X,
        'south' means crossing -X."""
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

            # Gate crossing check — walls are at constant X in SLAM frame
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

            # Arrival check
            if gate_dir is None and dist < DIST_TOL:
                self._arrival_clean = True
                break
            if gate_dir is None and dist < DIST_TOL * 2.0:
                self._arrival_clean = True
                break

            # Overshoot detection
            if gate_dir is None:
                if initial_dist is None:
                    initial_dist = dist
                if dist < min_dist:
                    min_dist = dist
                if min_dist < initial_dist * 0.75 and dist > min_dist + 0.04:
                    self.get_logger().info(
                        f'    DRV: overshoot detected — moving on')
                    break

            # Stuck detection — progress is along the gate axis
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

            # Speed control
            speed = LINEAR_SPEED
            if dist < 0.30:
                speed = max(0.04, LINEAR_SPEED * (dist / 0.30))
            # Slow down approaching a gate wall (distance in X)
            if gate_dir is not None:
                gate_dist = abs(tx - x)
                if gate_dist < 0.25:
                    speed = min(speed, max(0.04, LINEAR_SPEED * (gate_dist / 0.25)))
            if abs(yaw_err) > 0.5:
                speed *= 0.5

            # Lidar front slowdown
            front_min = self._get_front_min()
            if front_min < LIDAR_FRONT_STOP:
                speed = 0.0
                self.get_logger().warn(f'    EMERGENCY STOP: front={front_min:.2f}m')
            elif front_min < LIDAR_SLOW_DIST:
                speed *= max(0.3, front_min / LIDAR_SLOW_DIST)

            # Steering
            if abs(yaw_err) < STEER_DEADBAND:
                steer = 0.0
            else:
                steer = max(-MAX_STEER, min(MAX_STEER, yaw_err * STEER_GAIN))

            # Side wall correction
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

        # Skip if already past a gate
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
            if len(wp) == 3:
                wx, wy, gate_dir = wp
            else:
                wx, wy = wp
                gate_dir = None
            self.goto(wx, wy, label=f'[{label} {i+1}/{total}]',
                      gate_dir=gate_dir, recalibrate=recalibrate)
        self.get_logger().info(f'{label} route complete!')


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    nav = DirectNavigator()

    nav.get_logger().info('=== Left Coastal Navigator (SLAM Real Map) ===')
    nav.get_logger().info(f'Start: ({START_X:.3f},{START_Y:.3f}) yaw={START_YAW:.1f}')
    nav.get_logger().info(f'Port:  ({PORT_X:.3f},{PORT_Y:.3f})')
    nav.get_logger().info(f'Gaps:  wall1 y={GAP_Y[0]:.3f}  wall2 y={GAP_Y[1]:.3f}  '
                          f'wall3 y={GAP_Y[2]:.3f}')

    # Publish initial pose for AMCL (SLAM frame: 0,0 facing +X)
    nav.publish_initial_pose(START_X, START_Y, START_YAW)
    time.sleep(3.0)
    nav.wait_for_tf()

    # Verify AMCL convergence
    nav.verify_amcl_converged(START_X, START_Y, START_YAW)

    # Calibrate yaw offset
    nav.calibrate_yaw_offset()

    # ═════════════════════════════════════════════════════════════════
    # PHASE 1: DELIVER — Forward through zigzag to port
    # ═════════════════════════════════════════════════════════════════
    fwd = build_forward_route()
    nav.get_logger().info(f'\n>>> PHASE 1: DELIVER to port ({len(fwd)} waypoints)')
    nav.follow_route(fwd, label='FWD', recalibrate=False)
    nav.get_logger().info('--- Reached target port! ---')

    # ═════════════════════════════════════════════════════════════════
    # PHASE 2: DROP — Deliver starting cargo
    # ═════════════════════════════════════════════════════════════════
    nav.get_logger().info('\n>>> PHASE 2: DROP cargo at port')
    time.sleep(1.0)

    # ═════════════════════════════════════════════════════════════════
    # PHASE 3: PICKUP — Back up, pick up target cargo, return to port
    # ═════════════════════════════════════════════════════════════════
    nav.get_logger().info('\n>>> PHASE 3: PICKUP target cargo')
    pickup_pose = nav._spin_get_pose()
    if pickup_pose:
        # Back up slightly in -X (south) to pick up cargo
        back_x = pickup_pose[0] - 0.4
        nav.goto(pickup_pose[0], pickup_pose[1], label='[CARGO stay]',
                 gate_dir=None, recalibrate=False)
        nav.goto(back_x, pickup_pose[1], label='[CARGO backup]',
                 gate_dir=None, recalibrate=False)
        nav.goto(PORT_X, PORT_Y, label='[CARGO return]',
                 gate_dir=None, recalibrate=False)
    nav.get_logger().info('Target cargo picked up')

    # ═════════════════════════════════════════════════════════════════
    # PHASE 4: RETURN — Navigate home through zigzag
    # ═════════════════════════════════════════════════════════════════
    ret = build_return_route()
    nav.get_logger().info(f'\n>>> PHASE 4: RETURN to start ({len(ret)} waypoints)')
    nav.follow_route(ret, label='RET', recalibrate=False)

    # ═════════════════════════════════════════════════════════════════
    # DONE
    # ═════════════════════════════════════════════════════════════════
    nav.get_logger().info('=== Challenge complete! ===')
    nav._stop()
    nav.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
