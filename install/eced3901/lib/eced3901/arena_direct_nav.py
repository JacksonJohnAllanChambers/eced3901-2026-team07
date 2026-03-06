#! /usr/bin/env python3
"""
ECED3901 Challenge Arena – Direct cmd_vel Navigator
====================================================
Drives a known route using rotate-in-place then drive-straight segments.
No Nav2 planner/controller needed — just AMCL for map→odom alignment.

Robot pose is read from the TF tree (map → base_footprint) so waypoints
can be specified directly in map coordinates.

Strategy:
  1. Rotate in place to face the next waypoint
  2. Drive straight to it at constant speed
  3. Repeat for each waypoint

Usage (launch arena_gazebo + arena_amcl first):
  ros2 run eced3901 arena_direct_nav.py
  ros2 run eced3901 arena_direct_nav.py --ros-args -p start:=left_coastal
  ros2 run eced3901 arena_direct_nav.py --ros-args -p start:=right_coastal
  ros2 run eced3901 arena_direct_nav.py --ros-args -p start:=left_open
  ros2 run eced3901 arena_direct_nav.py --ros-args -p start:=right_open
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import tf2_ros

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
FT = 0.3048

NORTH = math.pi / 2
SOUTH = -math.pi / 2
EAST  = 0.0
WEST  = math.pi

# Speeds
LINEAR_SPEED  = 0.1     # m/s – constant forward speed (slow = stable)
ROTATE_SPEED  = 0.2      # rad/s – max rotation speed

# Tolerances
YAW_TOL       = 0.05     # rad (~5.7°) – heading tolerance before driving
DIST_TOL      = 0.06     # m – "arrived at waypoint" tolerance

# Driving behaviour
STEER_GAIN    = 1.2      # angular correction gain during driving
MAX_STEER     = 0.5      # rad/s – max angular correction while driving
STUCK_TIME    = 4.0      # seconds without odom movement → skip waypoint
GOTO_TIMEOUT  = 25.0     # seconds max per goto() call

# Control loop
DT = 1.0 / 30.0          # 30 Hz control loop

# ──────────────────────────────────────────────────────────────────────
# Start positions and port targets
# ──────────────────────────────────────────────────────────────────────
POSITIONS = {
    'left_coastal': {
        'start': (3*FT, 1.5*FT, NORTH),
        'port':  (3*FT, 12.5*FT),
    },
    'left_open': {
        'start': (5*FT, 1*FT, NORTH),
        'port':  (5*FT, 13*FT),
    },
    'right_open': {
        'start': (9*FT, 1*FT, NORTH),
        'port':  (9*FT, 13*FT),
    },
    'right_coastal': {
        'start': (11*FT, 1*FT, NORTH),
        'port':  (11*FT, 12.5*FT),
    },
}

# ──────────────────────────────────────────────────────────────────────
# Gap geometry – centres of the open gaps beside each zigzag wall
#
# Zigzag walls are 2 ft (0.610 m) long; gaps are 2 ft (0.610 m).
# Left coastal (x = 0 to 4ft = 0 to 1.219 m)
#   Wall from LEFT  → gap on RIGHT near divider, centre ≈ 0.88 m
#   Wall from RIGHT → gap on LEFT  near outer,   centre ≈ 0.34 m
# Right coastal (x = 10ft to 14ft = 3.048 to 4.267 m)
#   Wall from RIGHT → gap on LEFT  near divider, centre ≈ 3.39 m
#   Wall from LEFT  → gap on RIGHT near outer,   centre ≈ 3.93 m
# ──────────────────────────────────────────────────────────────────────
# Gap centres computed from wall collision geometry:
# All perimeter/divider walls 0.15 m thick; zigzag walls 0.610 m long.
# Left lane: outer-wall inner face x = 0.075, divider inner face x = 1.144
# Right lane: divider inner face x = 3.123, outer-wall inner face x = 4.192

# Gap centres pushed toward lane centre for extra wall clearance.
# Physical gap is 0.534 m wide; we aim ~0.10 m toward centre from true centre.
LC_GAP_R = 0.840  # gap near divider, pushed slightly left of (0.610+1.144)/2
LC_GAP_L = 0.420  # gap near outer wall, pushed right from 0.342 for margin
LC_MID   = 0.610  # lane centre

RC_GAP_L = 3.427  # gap near divider, pushed slightly right from 3.390
RC_GAP_R = 3.847  # gap near outer wall, pushed left from 3.925 for margin
RC_MID   = 3.658  # lane centre


# ──────────────────────────────────────────────────────────────────────
# Route builders
# ──────────────────────────────────────────────────────────────────────
def left_coastal_forward():
    """Waypoints to navigate north through left coastal zigzag.

    Zigzag walls are 2 ft long with 2 ft gaps.
    Wall 1 (y=4ft from left):  gap at x = R (near divider)
    Wall 2 (y=7ft from right): gap at x = L (near outer wall)
    Wall 3 (y=10ft from left): gap at x = R (near divider)

    Route uses gentle diagonals (~25° turns) between gaps instead of
    large rotations.  Pattern: straight→diagonal→straight→diagonal...
    """
    R, L = LC_GAP_R, LC_GAP_L
    return [
        # -- approach wall 1 gap (right side) heading north --
        (R, 0.85),       # line up at gap-R, well before wall 1
        (R, 1.45),       # straight north through wall 1, cleared
        # -- gentle diagonal to wall 2 gap (left side) --
        (L, 1.90),       # diagonal; 0.16m clearance below wall 2 (y=2.059)
        (L, 2.40),       # straight north through wall 2, cleared
        # -- gentle diagonal to wall 3 gap (right side) --
        (R, 2.75),       # diagonal; 0.22m clearance below wall 3 (y=2.973)
        (R, 3.30),       # straight north through wall 3, cleared
    ]

def left_coastal_return():
    """Waypoints to navigate south back through left coastal zigzag."""
    R, L = LC_GAP_R, LC_GAP_L
    return [
        # -- approach wall 3 gap heading south --
        (R, 3.30),       # align at gap-R, above wall 3
        (R, 2.75),       # south through wall 3, cleared
        # -- gentle diagonal to wall 2 gap --
        (L, 2.40),       # diagonal
        (L, 1.90),       # south through wall 2, cleared
        # -- gentle diagonal to wall 1 gap --
        (R, 1.45),       # diagonal
        (R, 0.75),       # south through wall 1, cleared
    ]

def right_coastal_forward():
    """Waypoints to navigate north through right coastal zigzag.

    Zigzag walls are 2 ft long with 2 ft gaps.
    Wall 1 (y=4ft from right):  gap at x = L (near divider)
    Wall 2 (y=7ft from left):   gap at x = R (near outer wall)
    Wall 3 (y=10ft from right): gap at x = L (near divider)

    Route uses gentle diagonals (~25° turns) between gaps.
    """
    L, R = RC_GAP_L, RC_GAP_R
    return [
        # -- approach wall 1 gap (left side) heading north --
        (L, 0.85),       # line up at gap-L, well before wall 1
        (L, 1.45),       # straight north through wall 1, cleared
        # -- gentle diagonal to wall 2 gap (right side) --
        (R, 1.90),       # diagonal; 0.16m clearance below wall 2
        (R, 2.40),       # straight north through wall 2, cleared
        # -- gentle diagonal to wall 3 gap (left side) --
        (L, 2.75),       # diagonal; 0.22m clearance below wall 3
        (L, 3.30),       # straight north through wall 3, cleared
    ]

def right_coastal_return():
    """Waypoints to navigate south back through right coastal zigzag."""
    L, R = RC_GAP_L, RC_GAP_R
    return [
        # -- approach wall 3 gap heading south --
        (L, 3.30),       # align at gap-L, above wall 3
        (L, 2.75),       # south through wall 3, cleared
        # -- gentle diagonal to wall 2 gap --
        (R, 2.40),       # diagonal
        (R, 1.90),       # south through wall 2, cleared
        # -- gentle diagonal to wall 1 gap --
        (L, 1.45),       # diagonal
        (L, 0.85),       # south through wall 1, cleared
    ]

def open_water_forward(start_x):
    """Straight north through open waters."""
    return [
        (start_x, 3*FT),
        (start_x, 6*FT),
        (start_x, 9*FT),
        (start_x, 12.0*FT),
    ]

def open_water_return(start_x):
    """Straight south through open waters."""
    return [
        (start_x, 9*FT),
        (start_x, 6*FT),
        (start_x, 3*FT),
        (start_x, 1.5*FT),
    ]


# ──────────────────────────────────────────────────────────────────────
# Navigator node
# ──────────────────────────────────────────────────────────────────────
class DirectNavigator(Node):
    def __init__(self):
        super().__init__('direct_navigator')

        self.declare_parameter('start', 'left_coastal')
        self.start_pos = self.get_parameter('start').get_parameter_value().string_value

        if self.start_pos not in POSITIONS:
            self.get_logger().error(f"Unknown start '{self.start_pos}'. "
                                    f"Options: {list(POSITIONS.keys())}")
            raise SystemExit(1)

        # cmd_vel publisher
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Initial pose publisher for AMCL
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        # Fused odom subscriber (EKF: wheel odom + IMU) for smooth heading
        self.last_odom = None
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Yaw offset: map_yaw = odom_yaw + yaw_offset
        # Computed once at startup after AMCL converges
        self.yaw_offset = 0.0

    def _odom_cb(self, msg):
        self.last_odom = msg

    # ─── Pose helpers ─────────────────────────────────────────────────

    def _get_odom_yaw(self):
        """Get smooth yaw from odom topic (no AMCL jitter)."""
        if self.last_odom is None:
            return None
        q = self.last_odom.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _get_odom_xy(self):
        """Get x, y from odom topic (continuous updates, no AMCL lag)."""
        if self.last_odom is None:
            return None
        p = self.last_odom.pose.pose.position
        return (p.x, p.y)

    def _get_map_yaw(self):
        """Get yaw using odom + calibrated offset. Smooth and map-aligned."""
        odom_yaw = self._get_odom_yaw()
        if odom_yaw is None:
            return None
        return self._normalize_angle(odom_yaw + self.yaw_offset)

    # ─── Pose from TF ────────────────────────────────────────────────

    def _get_pose(self):
        """Get robot position from map TF + smooth yaw from odom.
        Returns (x, y, yaw) or None."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
            x = t.transform.translation.x
            y = t.transform.translation.y
            # Use smooth odom-based yaw instead of jittery map TF yaw
            yaw = self._get_map_yaw()
            if yaw is None:
                # Fallback to TF yaw
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
        """Spin once and return pose. Returns (x, y, yaw) or None."""
        rclpy.spin_once(self, timeout_sec=0.05)
        return self._get_pose()

    # ─── Helpers ──────────────────────────────────────────────────────

    def _stop(self):
        self.cmd_pub.publish(Twist())

    @staticmethod
    def _normalize_angle(a):
        """Wrap angle to [-pi, pi]."""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    @staticmethod
    def _yaw_to_quat(yaw):
        """Return (qz, qw) for a 2D yaw."""
        return (math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    # ─── Setup ────────────────────────────────────────────────────────

    def calibrate_yaw_offset(self):
        """Compute yaw_offset = map_TF_yaw - odom_yaw so we can use
        smooth odom yaw for heading control while staying in map frame."""
        odom_yaw = self._get_odom_yaw()
        if odom_yaw is None:
            self.get_logger().warn('No odom data for yaw calibration')
            return
        # Get raw TF yaw (not our smoothed version)
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.5))
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            tf_yaw = math.atan2(siny, cosy)
        except Exception:
            self.get_logger().warn('No TF for yaw calibration')
            return
        self.yaw_offset = self._normalize_angle(tf_yaw - odom_yaw)
        self.get_logger().info(
            f'Yaw offset calibrated: odom_yaw={math.degrees(odom_yaw):.1f}° '
            f'+ offset={math.degrees(self.yaw_offset):.1f}° '
            f'= map_yaw={math.degrees(tf_yaw):.1f}°')

    def wait_for_tf(self, timeout=15.0):
        """Block until map→base_footprint TF is available."""
        self.get_logger().info('Waiting for map→base_footprint TF...')
        t0 = time.time()
        pose = None
        while pose is None:
            rclpy.spin_once(self, timeout_sec=0.2)
            pose = self._get_pose()
            if time.time() - t0 > timeout:
                self.get_logger().error('TF map→base_footprint not available!')
                raise SystemExit(1)
        self.get_logger().info(
            f'TF OK: ({pose[0]:.3f}, {pose[1]:.3f}) yaw={math.degrees(pose[2]):.1f}°')

    def _get_raw_tf_pose(self):
        """Get raw pose from map→base_footprint TF (including AMCL yaw, no odom smoothing)."""
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

    def verify_amcl_converged(self, expected_x, expected_y, expected_yaw, tol_xy=0.15, tol_yaw=0.3):
        """Verify AMCL has converged to near the expected initial pose.
        Uses raw TF yaw (not odom-based) since yaw_offset is not yet calibrated."""
        self.get_logger().info('Verifying AMCL convergence...')
        for attempt in range(30):  # up to 15 seconds
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.5)   # give AMCL time to process scans
            rclpy.spin_once(self, timeout_sec=0.1)
            pose = self._get_raw_tf_pose()
            if pose is None:
                continue
            x, y, yaw = pose
            dx = abs(x - expected_x)
            dy = abs(y - expected_y)
            dyaw = abs(self._normalize_angle(yaw - expected_yaw))
            self.get_logger().info(
                f'  AMCL check #{attempt+1}: pos=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):.1f}° '
                f'err=({dx:.3f},{dy:.3f},{math.degrees(dyaw):.1f}°)')
            if dx < tol_xy and dy < tol_xy and dyaw < tol_yaw:
                self.get_logger().info('  AMCL converged! ✓')
                return True
        self.get_logger().warn('AMCL may not have converged — proceeding anyway')
        return False

    def publish_initial_pose(self, x, y, yaw):
        """Tell AMCL where the robot starts."""
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        qz, qw = self._yaw_to_quat(yaw)
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance[0]  = 0.01   # x variance
        msg.pose.covariance[7]  = 0.01   # y variance
        msg.pose.covariance[35] = 0.005  # yaw variance
        for _ in range(5):
            self.initial_pose_pub.publish(msg)
            time.sleep(0.1)
        self.get_logger().info(
            f'Published initial pose: ({x:.3f}, {y:.3f}) yaw={math.degrees(yaw):.1f}°')

    # ─── Core movement primitives ─────────────────────────────────────

    def rotate_to(self, target_yaw):
        """Rotate in place until facing target_yaw (rad) in map frame."""
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
            speed = min(ROTATE_SPEED, max(0.06, abs(err) * 1.0))
            cmd.angular.z = speed if err > 0 else -speed
            cmd.linear.x = 0.0
            self.cmd_pub.publish(cmd)
            step += 1
            if step % 15 == 0:
                oy = self._get_odom_yaw()
                oy_str = f'{math.degrees(oy):.1f}' if oy else 'N/A'
                self.get_logger().info(
                    f'    ROT: yaw={math.degrees(yaw):.1f}° target={math.degrees(target_yaw):.1f}° '
                    f'err={math.degrees(err):.1f}° wz={cmd.angular.z:+.2f} odom_yaw={oy_str}°')
            time.sleep(DT)
        self._stop()

    def drive_to(self, tx, ty):
        """Drive toward (tx, ty) in map frame.
        Never re-rotates mid-drive. Uses proportional steering only.
        Detects stuck/overshoot and gives up gracefully."""
        cmd = Twist()
        step = 0
        min_dist = float('inf')
        last_progress_time = time.time()
        # Use odom position for stuck detection (updates continuously, not AMCL)
        odom_pos = self._get_odom_xy()
        last_odom_x = odom_pos[0] if odom_pos else 0.0
        last_odom_y = odom_pos[1] if odom_pos else 0.0

        while rclpy.ok():
            pose = self._spin_get_pose()
            if pose is None:
                time.sleep(DT)
                continue
            x, y, yaw = pose
            dx = tx - x
            dy = ty - y
            dist = math.hypot(dx, dy)

            # ---- Arrival check ----
            if dist < DIST_TOL:
                break

            # ---- Track minimum distance (overshoot detection) ----
            if dist < min_dist:
                min_dist = dist
            # If we're now 0.10m farther than closest approach, we overshot
            if dist > min_dist + 0.10:
                self.get_logger().info(
                    f'    DRV: overshoot detected (dist={dist:.3f} > min={min_dist:.3f}+0.10) — moving on')
                break

            # ---- Stuck detection via odom (continuous, immune to AMCL lag) ----
            now = time.time()
            odom_pos = self._get_odom_xy()
            if odom_pos:
                ox, oy = odom_pos
                odom_moved = math.hypot(ox - last_odom_x, oy - last_odom_y)
                if odom_moved > 0.02:   # >2cm of actual movement
                    last_odom_x, last_odom_y = ox, oy
                    last_progress_time = now
            if now - last_progress_time > STUCK_TIME:
                self.get_logger().warn(
                    f'    DRV: stuck for {STUCK_TIME:.1f}s at dist={dist:.3f} — moving on')
                break

            desired_yaw = math.atan2(dy, dx)
            yaw_err = self._normalize_angle(desired_yaw - yaw)

            step += 1
            if step % 20 == 0:
                oy = self._get_odom_yaw()
                oy_str = f'{math.degrees(oy):.1f}' if oy else 'N/A'
                self.get_logger().info(
                    f'    DRV: pos=({x:.3f},{y:.3f}) yaw={math.degrees(yaw):.1f}° '
                    f'target=({tx:.3f},{ty:.3f}) dist={dist:.3f} '
                    f'yaw_err={math.degrees(yaw_err):.1f}° odom_yaw={oy_str}°')

            # ---- Speed control ----
            # Ramp down near target
            speed = LINEAR_SPEED
            if dist < 0.15:
                speed = max(0.04, LINEAR_SPEED * (dist / 0.15))
            # Slow down when heading is off (>30° error → half speed)
            if abs(yaw_err) > 0.5:
                speed *= 0.5

            # ---- Steering: proportional, capped, NO re-rotation ----
            steer = max(-MAX_STEER, min(MAX_STEER, yaw_err * STEER_GAIN))

            cmd.linear.x = speed
            cmd.angular.z = steer
            self.cmd_pub.publish(cmd)
            time.sleep(DT)
        self._stop()

    def goto(self, tx, ty, label=''):
        """Rotate to face (tx,ty) then drive straight to it.
        Enforces a timeout to prevent getting stuck forever."""
        t0 = time.time()
        pose = self._spin_get_pose()
        if pose is None:
            self.get_logger().warn('No TF pose – skipping waypoint')
            return
        x, y, _ = pose
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        if dist < DIST_TOL:
            return
        target_yaw = math.atan2(dy, dx)

        self.get_logger().info(
            f'{label}  Rotate → {math.degrees(target_yaw):.0f}° '
            f'then drive {dist:.2f} m to ({tx:.3f}, {ty:.3f})')

        self.rotate_to(target_yaw)
        time.sleep(0.20)       # brief settle

        # Check timeout before driving
        if time.time() - t0 > GOTO_TIMEOUT:
            self.get_logger().warn(f'{label}  Timeout after rotation — moving on')
            self._stop()
            return

        self.drive_to(tx, ty)
        time.sleep(0.05)

    def follow_route(self, waypoints, label=''):
        """Follow a list of (x, y) waypoints using rotate→drive segments."""
        total = len(waypoints)
        for i, (wx, wy) in enumerate(waypoints):
            self.goto(wx, wy, label=f'[{label} {i+1}/{total}]')
        self.get_logger().info(f'{label} route complete!')


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    nav = DirectNavigator()

    pos = POSITIONS[nav.start_pos]
    sx, sy, syaw = pos['start']
    px, py = pos['port']
    is_coastal = 'coastal' in nav.start_pos

    nav.get_logger().info('=== Arena Direct Navigator ===')
    nav.get_logger().info(f'Start: {nav.start_pos}  deploy=({sx:.3f},{sy:.3f})  '
                          f'port=({px:.3f},{py:.3f})')

    # Publish initial pose for AMCL, then wait for TF to be ready
    nav.publish_initial_pose(sx, sy, syaw)
    time.sleep(3.0)  # let AMCL converge
    nav.wait_for_tf()

    # Verify AMCL is actually at the right pose before driving
    nav.verify_amcl_converged(sx, sy, syaw)

    # Calibrate yaw offset: map_yaw = odom_yaw + offset
    # This lets us use smooth odom yaw for heading control
    nav.calibrate_yaw_offset()

    # ── Build forward route ───────────────────────────────────────────
    if is_coastal:
        if 'left' in nav.start_pos:
            fwd = left_coastal_forward()
        else:
            fwd = right_coastal_forward()
    else:
        fwd = open_water_forward(sx)
    fwd.append((px, py))   # final: port centre

    # ── Navigate forward ──────────────────────────────────────────────
    nav.get_logger().info(f'\n>>> FORWARD to port ({len(fwd)} waypoints)')
    nav.follow_route(fwd, label='FWD')
    nav.get_logger().info('--- Reached target port! ---')

    # Pause at port (cargo ops would go here)
    time.sleep(1.0)

    # ── Build return route ────────────────────────────────────────────
    if is_coastal:
        if 'left' in nav.start_pos:
            ret = left_coastal_return()
        else:
            ret = right_coastal_return()
    else:
        ret = open_water_return(sx)
    ret.append((sx, sy))   # final: home deploy zone

    # ── Navigate return ───────────────────────────────────────────────
    nav.get_logger().info(f'\n>>> RETURN to deploy zone ({len(ret)} waypoints)')
    nav.follow_route(ret, label='RET')
    nav.get_logger().info('=== Challenge complete! ===')

    nav._stop()
    nav.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
