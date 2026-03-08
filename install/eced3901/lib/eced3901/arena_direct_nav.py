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
  ros2 run eced3901 arena_direct_nav.py                          # auto-detect position
  ros2 run eced3901 arena_direct_nav.py --ros-args -p start:=auto
  ros2 run eced3901 arena_direct_nav.py --ros-args -p start:=left_coastal
  ros2 run eced3901 arena_direct_nav.py --ros-args -p start:=right_coastal
  ros2 run eced3901 arena_direct_nav.py --ros-args -p start:=left_open
  ros2 run eced3901 arena_direct_nav.py --ros-args -p start:=right_open
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from gazebo_msgs.srv import GetEntityState, DeleteEntity
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import String
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
LINEAR_SPEED  = 0.25     # m/s – forward speed (IMU heading keeps accuracy at speed)
ROTATE_SPEED  = 0.5      # rad/s – max rotation speed

# Tolerances
YAW_TOL       = 0.05     # rad (~2.9°) – heading tolerance before driving
DIST_TOL      = 0.10     # m – "arrived at waypoint" tolerance

# Driving behaviour
STEER_GAIN    = 1.0      # angular correction gain during driving
MAX_STEER     = 0.5      # rad/s – max angular correction while driving
STEER_DEADBAND = 0.03    # rad – below this heading error, no angular correction
STUCK_TIME    = 4.0      # seconds without goal progress → skip waypoint
GOTO_TIMEOUT  = 20.0     # seconds max per goto() call

# Lidar safety
LIDAR_SIDE_WARN   = 0.10   # m – steer correction when wall closer than this
                           # (set below lidar range_min ~0.12 to avoid spurious
                           #  corrections from wall-edge glancing readings)
LIDAR_SIDE_GAIN   = 3.0    # steering correction per metre of side closeness
LIDAR_SLOW_DIST   = 0.40   # m – start slowing when obstacle closer than this

# Control loop
DT = 1.0 / 30.0          # 30 Hz control loop

# ──────────────────────────────────────────────────────────────────────
# Start positions and port targets
# ──────────────────────────────────────────────────────────────────────
POSITIONS = {
    'left_coastal': {
        'start': (3*FT, 1*FT, NORTH),
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
# Gap geometry – TRUE centres of the open gaps (from world file)
#
# All perimeter/divider walls 0.15 m thick; zigzag walls 0.610 m long × 0.15 m thick.
# Left lane:  outer-wall inner face x = 0.075, divider inner face x = 1.144
# Right lane: divider inner face x = 3.123, outer-wall inner face x = 4.192
# ──────────────────────────────────────────────────────────────────────
# Walls 1 & 3 extend from LEFT  wall → gap on RIGHT: (0.610 + 1.144)/2 = 0.877
# Wall 2      extends from RIGHT wall → gap on LEFT:  (0.075 + 0.609)/2 = 0.342
LC_GAP_R = 0.877  # left coastal, gap near divider (walls 1 & 3)
LC_GAP_L = 0.342  # left coastal, gap near outer wall (wall 2)

# Right coastal mirrors: walls 1 & 3 from RIGHT → gap LEFT, wall 2 from LEFT → gap RIGHT
RC_GAP_L = 3.390  # right coastal, gap near divider (walls 1 & 3)
RC_GAP_R = 3.925  # right coastal, gap near outer wall (wall 2)

# Zigzag wall y-positions
WALL_Y = (4*FT, 7*FT, 10*FT)  # (1.219, 2.134, 3.048)


# ──────────────────────────────────────────────────────────────────────
# Route builders
#
# Waypoints are either:
#   (x, y)             – destination: arrive within DIST_TOL
#   (x, y, 'north')    – gate: robot passes northward through wall at y
#   (x, y, 'south')    – gate: robot passes southward through wall at y
#
# Gate waypoints fire as soon as the robot's y crosses the wall y-level,
# so the robot never orbits or stops at the gap — it just drives through.
# ──────────────────────────────────────────────────────────────────────
def left_coastal_forward():
    """Gate waypoints at true gap centres — northbound."""
    R, L = LC_GAP_R, LC_GAP_L
    return [
        (R, WALL_Y[0], 'north'),  # through wall 1 gap (near divider)
        (L, WALL_Y[1], 'north'),  # through wall 2 gap (near outer wall)
        (R, WALL_Y[2], 'north'),  # through wall 3 gap (near divider)
    ]

def left_coastal_return():
    """Gate waypoints at true gap centres — southbound.
    Diagonals broken into south→lateral segments to avoid oblique
    headings where lidar glancing causes heading instability."""
    R, L = LC_GAP_R, LC_GAP_L
    mid_32 = (WALL_Y[2] + WALL_Y[1]) / 2   # midpoint y between wall 3 and wall 2
    mid_21 = (WALL_Y[1] + WALL_Y[0]) / 2   # midpoint y between wall 2 and wall 1
    return [
        (R, WALL_Y[2], 'south'),  # through wall 3 gap
        (R, mid_32),              # south to safe y at gap-3 x
        (L, mid_32),              # lateral to gap-2 x
        (L, WALL_Y[1], 'south'),  # through wall 2 gap
        (L, mid_21),              # south to safe y at gap-2 x
        (R, mid_21),              # lateral to gap-1 x
        (R, WALL_Y[0], 'south'),  # through wall 1 gap
    ]

def right_coastal_forward():
    """Gate waypoints at true gap centres — northbound."""
    L, R = RC_GAP_L, RC_GAP_R
    return [
        (L, WALL_Y[0], 'north'),  # through wall 1 gap (near divider)
        (R, WALL_Y[1], 'north'),  # through wall 2 gap (near outer wall)
        (L, WALL_Y[2], 'north'),  # through wall 3 gap (near divider)
    ]

def right_coastal_return():
    """Gate waypoints at true gap centres — southbound.
    Diagonals broken into south→lateral segments."""
    L, R = RC_GAP_L, RC_GAP_R
    mid_32 = (WALL_Y[2] + WALL_Y[1]) / 2
    mid_21 = (WALL_Y[1] + WALL_Y[0]) / 2
    return [
        (L, WALL_Y[2], 'south'),  # through wall 3 gap
        (L, mid_32),              # south to safe y at gap-3 x
        (R, mid_32),              # lateral to gap-2 x
        (R, WALL_Y[1], 'south'),  # through wall 2 gap
        (R, mid_21),              # south to safe y at gap-2 x
        (L, mid_21),              # lateral to gap-1 x
        (L, WALL_Y[0], 'south'),  # through wall 1 gap
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

        self.declare_parameter('start', 'auto')
        self.start_pos = self.get_parameter('start').get_parameter_value().string_value

        valid = list(POSITIONS.keys()) + ['auto']
        if self.start_pos not in valid:
            self.get_logger().error(f"Unknown start '{self.start_pos}'. "
                                    f"Options: {valid}")
            raise SystemExit(1)

        # cmd_vel publisher
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Initial pose publisher for AMCL
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        # Fused odom subscriber (raw diff_drive odom for smooth position/heading)
        self.last_odom = None
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        # Lidar subscriber (for reactive wall avoidance)
        self.last_scan = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        # IMU subscriber (for drift-free heading)
        self.last_imu = None
        self.create_subscription(Imu, '/imu/data', self._imu_cb, 10)

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Gazebo ground-truth service client
        self.gz_state_cli = self.create_client(
            GetEntityState, '/gazebo/get_entity_state')
        self._gz_available = None  # None=unknown, True/False after first check

        # ── CV pickup system ───────────────────────────────────────
        self.declare_parameter('enable_cv_inspection', True)
        self._cv_enabled = self.get_parameter('enable_cv_inspection').value
        self._cv_picked_up = set()    # model_names already picked up
        self._latest_cv = None        # latest detection dict
        self._cv_phase = 'off'        # 'off', 'deliver', 'pickup_cargo', 'return'
        self._cv_target_types = set() # which object types to react to
        self._cv_path_suffix = ''     # e.g. 'left_coastal' — only react to own path
        self.create_subscription(String, '/cv/detections', self._cv_cb, 10)

        # Gazebo delete service (for simulated pickup)
        self.gz_delete_cli = self.create_client(DeleteEntity, '/delete_entity')

        # Yaw offset: map_yaw = odom_yaw + yaw_offset (used for XY transform)
        # Computed once at startup after AMCL converges
        self.yaw_offset = 0.0

        # IMU yaw offset: map_yaw = imu_yaw + imu_yaw_offset (used for heading)
        self.imu_yaw_offset = 0.0

        # Position offsets: map_xy = R(yaw_offset) @ odom_xy + (x_off, y_off)
        self.x_offset = 0.0
        self.y_offset = 0.0

    def _odom_cb(self, msg):
        self.last_odom = msg

    def _scan_cb(self, msg):
        self.last_scan = msg

    def _imu_cb(self, msg):
        self.last_imu = msg

    def _cv_cb(self, msg):
        """Callback for /cv/detections (JSON string)."""
        try:
            det = json.loads(msg.data)
            self._latest_cv = det
            if self._cv_phase not in ('off', 'deliver'):
                self.get_logger().info(
                    f'  CV rx: {det.get("type")} {det.get("model_name")} '
                    f'd={det.get("distance",0):.2f}m '
                    f'b={det.get("bearing",0):.2f}rad '
                    f'c={det.get("confidence",0):.2f}')
        except json.JSONDecodeError:
            pass

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

    def _get_imu_yaw(self):
        """Get yaw from IMU orientation (drift-free in simulation)."""
        if self.last_imu is None:
            return None
        q = self.last_imu.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny, cosy)

    def _get_ground_truth(self):
        """Query Gazebo for the robot's true (x, y, yaw). Returns None if unavailable."""
        if self._gz_available is False:
            return None
        if not self.gz_state_cli.service_is_ready():
            if self._gz_available is None:
                self._gz_available = False
                self.get_logger().info('Ground-truth service not available (gazebo_ros_state plugin not loaded)')
            return None
        if self._gz_available is None:
            self._gz_available = True
            self.get_logger().info('Ground-truth service available ✓')
        req = GetEntityState.Request()
        req.name = 'eced3901bot'
        req.reference_frame = 'world'
        future = self.gz_state_cli.call_async(req)
        # Spin until we get a response (with short timeout)
        t0 = time.time()
        while not future.done() and time.time() - t0 < 0.3:
            rclpy.spin_once(self, timeout_sec=0.05)
        if not future.done():
            return None
        resp = future.result()
        if resp is None or not resp.success:
            return None
        p = resp.state.pose.position
        q = resp.state.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)
        return (p.x, p.y, yaw)

    def _log_ground_truth(self, label=''):
        """Log ground-truth pose vs estimated pose for diagnostics."""
        gt = self._get_ground_truth()
        if gt is None:
            return
        est = self._get_pose()
        if est is None:
            return
        gx, gy, gyaw = gt
        ex, ey, eyaw = est
        dx = gx - ex
        dy = gy - ey
        dyaw = self._normalize_angle(gyaw - eyaw)
        self.get_logger().info(
            f'  GT {label}: true=({gx:.3f},{gy:.3f},{math.degrees(gyaw):.1f}°) '
            f'est=({ex:.3f},{ey:.3f},{math.degrees(eyaw):.1f}°) '
            f'err=({dx:+.3f},{dy:+.3f},{math.degrees(dyaw):+.1f}°)')

    def _get_map_yaw(self):
        """Get yaw using IMU + calibrated offset (preferred, drift-free).
        Falls back to odom yaw if IMU is unavailable."""
        imu_yaw = self._get_imu_yaw()
        if imu_yaw is not None:
            return self._normalize_angle(imu_yaw + self.imu_yaw_offset)
        odom_yaw = self._get_odom_yaw()
        if odom_yaw is None:
            return None
        return self._normalize_angle(odom_yaw + self.yaw_offset)

    def _get_map_xy(self):
        """Get (x, y) in map frame using incremental odom tracking.
        Uses IMU-corrected rotation at each step so odom yaw drift
        doesn't corrupt the XY position over time."""
        odom_xy = self._get_odom_xy()
        odom_yaw = self._get_odom_yaw()
        if odom_xy is None:
            return None
        ox, oy = odom_xy

        if not hasattr(self, '_prev_odom_xy') or self._prev_odom_xy is None:
            # First call or after reset: use static transform
            c = math.cos(self.yaw_offset)
            s = math.sin(self.yaw_offset)
            self._map_x = ox * c - oy * s + self.x_offset
            self._map_y = ox * s + oy * c + self.y_offset
            self._prev_odom_xy = (ox, oy)
            return (self._map_x, self._map_y)

        # Odom displacement since last call
        dox = ox - self._prev_odom_xy[0]
        doy = oy - self._prev_odom_xy[1]
        self._prev_odom_xy = (ox, oy)

        # Get current odom→map rotation from IMU (drift-free)
        imu_yaw = self._get_imu_yaw()
        if imu_yaw is not None and odom_yaw is not None:
            cur_yaw_off = self._normalize_angle(
                imu_yaw + self.imu_yaw_offset - odom_yaw)
        else:
            cur_yaw_off = self.yaw_offset

        # Rotate odom increment to map frame and accumulate
        c = math.cos(cur_yaw_off)
        s = math.sin(cur_yaw_off)
        self._map_x += dox * c - doy * s
        self._map_y += dox * s + doy * c
        return (self._map_x, self._map_y)

    # ─── Pose from odom ──────────────────────────────────────────────

    def _get_raw_tf_xy(self):
        """Get raw (x, y) from map→base_footprint TF (for calibration)."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.3))
            return (t.transform.translation.x, t.transform.translation.y)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None

    def _get_pose(self):
        """Get robot position in map frame using odom-based tracking.
        Position: from raw odom + calibrated XY offset (smooth, no AMCL jitter).
        Yaw: from raw odom + calibrated yaw offset (smooth).
        Falls back to raw TF if odom data is unavailable."""
        map_xy = self._get_map_xy()
        map_yaw = self._get_map_yaw()
        if map_xy is not None and map_yaw is not None:
            return (map_xy[0], map_xy[1], map_yaw)
        # Fallback to raw TF
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
        """Spin once and return pose. Returns (x, y, yaw) or None.
        Retries a few times on transient TF failures."""
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)
            pose = self._get_pose()
            if pose is not None:
                return pose
            time.sleep(0.2)
        return None

    # ─── Lidar helpers ────────────────────────────────────────────────

    def _lidar_sector_min(self, start_deg, end_deg):
        """Min valid range in a degree sector of the lidar scan."""
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
        """Min range in front ±15° sector."""
        return min(self._lidar_sector_min(0, 15),
                   self._lidar_sector_min(345, 359))

    def _get_side_mins(self):
        """Min range on left (40°–120°) and right (240°–320°)."""
        return (self._lidar_sector_min(40, 120),
                self._lidar_sector_min(240, 320))

    def _get_rear_min(self):
        """Min range in rear ±15° sector (around 180°)."""
        return self._lidar_sector_min(165, 195)

    # ─── Auto-detection ───────────────────────────────────────────────

    def detect_position(self):
        """Auto-detect which of the 4 starting positions the robot is in
        by reading lidar left/right wall distances at startup.

        Uses narrow perpendicular sectors (85°-95° and 265°-275°) to get
        true side-wall distances without bottom wall contamination.
        (Wide sectors pick up the nearby bottom wall at oblique angles.)

        Robot faces north at y≈0.305m. The 4 positions have distinct signatures:
          left_coastal  (x=0.914): L≈0.84m  R≈0.23m  (corridor, close right)
          left_open     (x=1.524): L≈0.23m  R≈1.45m  (close left, far right)
          right_open    (x=2.743): L≈1.45m  R≈0.23m  (far left, close right)
          right_coastal (x=3.353): L≈0.23m  R≈0.84m  (corridor, close left)
        """
        self.get_logger().info('Auto-detecting starting position...')
        # Collect several scans for stability
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
        # Narrow perpendicular sectors to avoid bottom wall interference
        left_min = self._lidar_sector_min(85, 95)    # pure left (west)
        right_min = self._lidar_sector_min(265, 275)  # pure right (east)
        self.get_logger().info(
            f'  Lidar (perpendicular): left={left_min:.3f}m  right={right_min:.3f}m')

        # Classification: each position maps to a unique (close/far) pair
        # "close" ≈ 0.23m (divider wall), "mid" ≈ 0.84m (corridor), "far" ≈ 1.45m (open)
        CLOSE_THRESH = 0.5   # below = next to divider wall
        FAR_THRESH   = 1.0   # above = open water (no nearby wall)

        if left_min > FAR_THRESH and right_min < CLOSE_THRESH:
            detected = 'right_open'        # far left, close right
        elif left_min < CLOSE_THRESH and right_min > FAR_THRESH:
            detected = 'left_open'         # close left, far right
        elif left_min > CLOSE_THRESH and right_min < CLOSE_THRESH:
            detected = 'left_coastal'      # mid left (~0.84), close right
        elif left_min < CLOSE_THRESH and right_min > CLOSE_THRESH:
            detected = 'right_coastal'     # close left, mid right (~0.84)
        else:
            # Both similar — unusual, use ratio as tiebreaker
            self.get_logger().warn(
                f'Ambiguous lidar signature (L={left_min:.3f}, R={right_min:.3f})')
            if left_min > right_min:
                detected = 'left_coastal'
            else:
                detected = 'right_coastal'

        self.get_logger().info(f'  Detected position: {detected}')
        return detected

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
        """Compute yaw_offset and XY offsets so we can use smooth raw odom
        for heading and position control in map frame.
        map_yaw = odom_yaw + yaw_offset
        map_xy  = R(yaw_offset) @ odom_xy + (x_offset, y_offset)"""
        odom_yaw = self._get_odom_yaw()
        odom_xy = self._get_odom_xy()
        if odom_yaw is None or odom_xy is None:
            self.get_logger().warn('No odom data for calibration')
            return
        # Get raw TF pose (from AMCL — should be well-converged at this point)
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
        # offset = map_xy - R(yaw_offset) @ odom_xy
        ox, oy = odom_xy
        c = math.cos(self.yaw_offset)
        s = math.sin(self.yaw_offset)
        self.x_offset = tf_x - (ox * c - oy * s)
        self.y_offset = tf_y - (ox * s + oy * c)
        # Reset incremental position tracker
        self._prev_odom_xy = None
        # Also calibrate IMU yaw offset
        imu_yaw = self._get_imu_yaw()
        if imu_yaw is not None:
            self.imu_yaw_offset = self._normalize_angle(tf_yaw - imu_yaw)
            self.get_logger().info(
                f'Calibrated: imu_yaw={math.degrees(imu_yaw):.1f}° '
                f'+ offset={math.degrees(self.imu_yaw_offset):.1f}° '
                f'= map_yaw={math.degrees(tf_yaw):.1f}°  '
                f'xy_offset=({self.x_offset:.3f},{self.y_offset:.3f})')
        else:
            self.get_logger().info(
                f'Calibrated: odom_yaw={math.degrees(odom_yaw):.1f}° '
                f'+ offset={math.degrees(self.yaw_offset):.1f}° '
                f'= map_yaw={math.degrees(tf_yaw):.1f}°  '
                f'xy_offset=({self.x_offset:.3f},{self.y_offset:.3f}) [no IMU]')
        self._log_ground_truth('CALIBRATED')

    def wait_for_tf(self, timeout=30.0):
        """Block until map→base_footprint TF is available."""
        self.get_logger().info('Waiting for map→base_footprint TF...')
        t0 = time.time()
        pose = None
        while pose is None:
            # Spin aggressively to ensure TF subscription discovers publishers
            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=0.1)
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
        Re-publishes initial pose every 15 attempts to recover from AMCL ignoring the first one.
        Uses raw TF yaw (not odom-based) since yaw_offset is not yet calibrated."""
        self.get_logger().info('Verifying AMCL convergence...')
        for attempt in range(60):  # up to 30 seconds
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
            # Re-publish initial pose periodically in case AMCL missed the first one
            if attempt % 15 == 14:
                self.get_logger().info('  Re-publishing initial pose...')
                self.publish_initial_pose(expected_x, expected_y, expected_yaw)
        self.get_logger().error('AMCL failed to converge after 60 attempts!')
        self.get_logger().error('Check: is arena_amcl.launch.py running? Is the map loaded?')
        raise SystemExit(1)

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

    def drive_to(self, tx, ty, gate_dir=None):
        """Drive toward (tx, ty) in map frame.
        Uses proportional steering with lidar-based wall avoidance.
        Detects stuck/overshoot and gives up gracefully.

        gate_dir: None for destination waypoints (distance-based arrival).
                  'north' or 'south' for gate waypoints (y-crossing arrival).
        """
        self._arrival_clean = False  # set True only on clean arrival
        cmd = Twist()
        step = 0
        min_dist = float('inf')
        initial_dist = None
        last_progress_time = time.time()
        best_progress = float('-inf')  # tracks best progress toward goal

        while rclpy.ok():
            pose = self._spin_get_pose()
            if pose is None:
                time.sleep(DT)
                continue
            x, y, yaw = pose
            dx = tx - x
            dy = ty - y
            dist = math.hypot(dx, dy)

            # ---- Gate crossing check (passthrough waypoints) ----
            if gate_dir is not None:
                crossed = (y > ty) if gate_dir == 'north' else (y < ty)
                if crossed:
                    self.get_logger().info(
                        f'    GATE: crossed y={ty:.3f} at pos=({x:.3f},{y:.3f})')
                    self._arrival_clean = True
                    break

            # ---- Arrival check (destination waypoints) ----
            if gate_dir is None and dist < DIST_TOL:
                self._arrival_clean = True
                break

            # ---- Final approach: stop when very close (destination only) ----
            if gate_dir is None and dist < DIST_TOL * 2.0:
                self._arrival_clean = True
                break

            # ---- Track minimum distance (overshoot detection, destination only) ----
            if gate_dir is None:
                if initial_dist is None:
                    initial_dist = dist
                if dist < min_dist:
                    min_dist = dist
                # Only check overshoot after making meaningful progress (>25% of distance)
                if min_dist < initial_dist * 0.75 and dist > min_dist + 0.04:
                    self.get_logger().info(
                        f'    DRV: overshoot detected (dist={dist:.3f} > min={min_dist:.3f}+0.04) — moving on')
                    break

            # ---- Stuck detection via goal progress ----
            now = time.time()
            if gate_dir == 'north':
                progress = y
            elif gate_dir == 'south':
                progress = -y
            else:
                progress = -dist
            if progress > best_progress + 0.01:
                best_progress = progress
                last_progress_time = now
            if now - last_progress_time > STUCK_TIME:
                self.get_logger().warn(
                    f'    DRV: stuck for {STUCK_TIME:.1f}s (no goal progress) '
                    f'dist={dist:.3f} — moving on')
                break

            desired_yaw = math.atan2(dy, dx)
            yaw_err = self._normalize_angle(desired_yaw - yaw)

            step += 1
            if step % 20 == 0:
                f_r = self._get_front_min()
                l_r, r_r = self._get_side_mins()
                self.get_logger().info(
                    f'    DRV: pos=({x:.3f},{y:.3f}) dist={dist:.3f} '
                    f'yaw_err={math.degrees(yaw_err):.1f}° '
                    f'lidar F={f_r:.2f} L={l_r:.2f} R={r_r:.2f}')
            if step % 60 == 0:
                self._log_ground_truth('DRV')

            # ---- Speed control ----
            speed = LINEAR_SPEED
            if dist < 0.15:
                speed = max(0.04, LINEAR_SPEED * (dist / 0.15))
            if abs(yaw_err) > 0.5:
                speed *= 0.5

            # ---- Lidar safety: slow down near front obstacles ----
            front_min = self._get_front_min()
            if front_min < LIDAR_SLOW_DIST:
                speed *= max(0.3, front_min / LIDAR_SLOW_DIST)

            # ---- Steering: proportional with dead band, capped ----
            if abs(yaw_err) < STEER_DEADBAND:
                steer = 0.0
            else:
                steer = max(-MAX_STEER, min(MAX_STEER, yaw_err * STEER_GAIN))

            # ---- Lidar safety: side wall correction ----
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

            # Drain pending callbacks so CV detections are processed
            for _ in range(3):
                rclpy.spin_once(self, timeout_sec=0.001)

            # ── CV pickup check during driving ──
            if self._check_cv_pickup():
                # After pickup, re-acquire pose and continue
                pose = self._spin_get_pose()
                if pose is not None:
                    x, y, yaw = pose

            time.sleep(DT)
        self._stop()

    def goto(self, tx, ty, label='', gate_dir=None, recalibrate=True):
        """Rotate to face (tx,ty) then drive to it.
        gate_dir: None for destination, 'north'/'south' for gate passthrough.
        recalibrate: If True, recalibrate yaw on clean dest arrival."""
        t0 = time.time()
        pose = self._spin_get_pose()
        if pose is None:
            self.get_logger().warn('No TF pose – skipping waypoint')
            return
        x, y, _ = pose

        # For gates, check if already past the wall
        if gate_dir is not None:
            already_through = (y > ty) if gate_dir == 'north' else (y < ty)
            if already_through:
                self.get_logger().info(f'{label}  Already past gate y={ty:.3f} (cur y={y:.3f}) — skipping')
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
            f'{label}  [{kind}] Rotate → {math.degrees(target_yaw):.0f}° '
            f'then drive {dist:.2f} m to ({tx:.3f}, {ty:.3f})')
        self._log_ground_truth(f'{label} START')

        self.rotate_to(target_yaw)
        time.sleep(0.20)       # brief settle

        # Check timeout before driving
        if time.time() - t0 > GOTO_TIMEOUT:
            self.get_logger().warn(f'{label}  Timeout after rotation — moving on')
            self._stop()
            return

        self.drive_to(tx, ty, gate_dir=gate_dir)
        time.sleep(0.05)

        # Recalibrate yaw offset on clean arrival at DESTINATIONS only
        # (gates are passthrough — recalibrating mid-run with jittery AMCL is risky)
        if gate_dir is None and recalibrate and getattr(self, '_arrival_clean', False):
            self._recalibrate_yaw_offset()
        elif gate_dir is None:
            self.get_logger().info('Skipping yaw recalibration (unclean arrival)')

    def _recalibrate_yaw_offset(self):
        """Re-read map TF and recompute yaw and XY offsets to correct for drift.
        Rejects large yaw changes (>8°) and XY jumps (>0.3m) to protect
        against AMCL particle filter teleports."""
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
                f'Recalibration rejected: yaw change {math.degrees(delta):.1f}° > 8° '
                f'(AMCL jitter likely). Keeping offset {math.degrees(old_offset):.1f}°')
            return
        # Check XY sanity: reject AMCL teleports
        cur_xy = self._get_map_xy()
        if cur_xy is not None:
            xy_jump = math.hypot(tf_x - cur_xy[0], tf_y - cur_xy[1])
            if xy_jump > 0.3:
                self.get_logger().warn(
                    f'Recalibration rejected: XY jump {xy_jump:.3f}m > 0.3m '
                    f'(AMCL teleport likely). Keeping current position.')
                return
        self.yaw_offset = new_offset
        ox, oy = odom_xy
        c = math.cos(self.yaw_offset)
        s = math.sin(self.yaw_offset)
        self.x_offset = tf_x - (ox * c - oy * s)
        self.y_offset = tf_y - (ox * s + oy * c)
        # Reset incremental position tracker
        self._prev_odom_xy = None
        # NOTE: do NOT update imu_yaw_offset here — IMU heading is drift-free
        # and AMCL yaw jitter would corrupt it. Only XY offsets and yaw_offset
        # (used for the odom→map rotation of XY) are updated.
        self.get_logger().info(
            f'Recalibrated yaw offset: {math.degrees(old_offset):.1f}° -> '
            f'{math.degrees(self.yaw_offset):.1f}°')
        self._log_ground_truth('RECALIBRATED')

    # ─── CV pickup behaviour ─────────────────────────────────────

    def set_cv_phase(self, phase, target_types=None):
        """Set the current mission phase for CV filtering.
        phase: 'off', 'deliver', 'pickup_cargo', 'return'
        target_types: set of object types to react to, e.g. {'cargo'}, {'lifeboat'}
        """
        self._cv_phase = phase
        self._cv_target_types = target_types or set()
        self.get_logger().info(
            f'CV phase: {phase}  targets: {self._cv_target_types}')

    def _delete_gazebo_model(self, model_name):
        """Delete a model from Gazebo (simulates picking it up)."""
        if not self.gz_delete_cli.service_is_ready():
            self.get_logger().warn(f'Delete service not ready, cannot remove {model_name}')
            return False
        req = DeleteEntity.Request()
        req.name = model_name
        future = self.gz_delete_cli.call_async(req)
        t0 = time.time()
        while not future.done() and time.time() - t0 < 3.0:
            rclpy.spin_once(self, timeout_sec=0.05)
        if future.done() and future.result() is not None and future.result().success:
            self.get_logger().info(f'Removed {model_name} from Gazebo (picked up)')
            return True
        self.get_logger().warn(f'Failed to remove {model_name}')
        return False

    def _cv_approach_and_pickup(self, det):
        """CV-guided approach: steer toward the detected object until close,
        then simulate pickup (spin + delete from Gazebo).
        Returns True if pickup was performed."""
        model_name = det.get('model_name', '')
        obj_type = det.get('type', 'unknown')
        distance = det.get('distance', 999.0)
        bearing = det.get('bearing', 0.0)

        self.get_logger().info(
            f'CV detected {obj_type} ({model_name}) at {distance:.2f}m, '
            f'bearing {bearing:.3f} rad — approaching for pickup')

        # Phase 1: approach the object using CV bearing
        APPROACH_SPEED = 0.12    # m/s — slow approach
        APPROACH_TOL = 0.25      # m — close enough to "grab"
        APPROACH_TIMEOUT = 15.0  # s
        cmd = Twist()
        t0 = time.time()

        while rclpy.ok() and (time.time() - t0) < APPROACH_TIMEOUT:
            rclpy.spin_once(self, timeout_sec=0.05)

            # Use latest CV detection if available
            cv = self._latest_cv
            if cv and cv.get('model_name') == model_name:
                distance = cv['distance']
                bearing = cv['bearing']

            if distance < APPROACH_TOL:
                self.get_logger().info(
                    f'  Close enough ({distance:.2f}m) — initiating pickup')
                break

            # Steer toward object: proportional bearing correction
            cmd.linear.x = APPROACH_SPEED
            cmd.angular.z = max(-0.4, min(0.4, bearing * 1.5))
            self.cmd_pub.publish(cmd)
            time.sleep(DT)

        self._stop()
        time.sleep(0.2)

        # Phase 2: simulate pickup action (slow spin, like an arm mechanism)
        PICKUP_SPEED = 0.4    # rad/s
        PICKUP_ANGLE = math.pi  # 180° spin simulates pickup action
        spin_duration = PICKUP_ANGLE / PICKUP_SPEED
        self.get_logger().info(
            f'  Simulating pickup of {obj_type} ({spin_duration:.1f}s spin)...')
        cmd = Twist()
        cmd.angular.z = PICKUP_SPEED
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < spin_duration:
            self.cmd_pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(DT)
        self._stop()
        time.sleep(0.2)

        # Phase 3: delete object from Gazebo
        self._delete_gazebo_model(model_name)
        self._cv_picked_up.add(model_name)
        self.get_logger().info(
            f'Pickup complete: {obj_type} ({model_name})')
        return True

    def _check_cv_pickup(self):
        """Check if there's a CV detection we should react to in the
        current mission phase.  Returns True if a pickup was performed."""
        if not self._cv_enabled or self._cv_phase == 'off':
            return False
        det = self._latest_cv
        if det is None:
            return False
        obj_type = det.get('type', '')
        model_name = det.get('model_name', '')
        confidence = det.get('confidence', 0.0)
        distance = det.get('distance', 999.0)

        # Filter: only react to target types for current phase
        if obj_type not in self._cv_target_types:
            return False
        if model_name in self._cv_picked_up:
            return False
        # Only react to objects on our own path
        if self._cv_path_suffix and self._cv_path_suffix not in model_name:
            return False
        if confidence <= 0.5 or distance >= 1.5:
            return False

        return self._cv_approach_and_pickup(det)

    def follow_route(self, waypoints, label='', recalibrate=True):
        """Follow a list of waypoints.
        Each wp is (x, y) for destinations or (x, y, 'north'/'south') for gates.
        recalibrate: If True, recalibrate yaw at destination waypoints."""
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


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    rclpy.init()
    nav = DirectNavigator()

    # ── Auto-detect or use specified position ─────────────────────────
    if nav.start_pos == 'auto':
        # Need lidar data for detection — spin briefly to populate
        for _ in range(30):
            rclpy.spin_once(nav, timeout_sec=0.1)
        nav.start_pos = nav.detect_position()

    pos = POSITIONS[nav.start_pos]
    sx, sy, syaw = pos['start']
    px, py = pos['port']
    is_coastal = 'coastal' in nav.start_pos

    # Set CV path filter so only objects on our own path are picked up
    nav._cv_path_suffix = nav.start_pos  # e.g. 'left_coastal'

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

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: DELIVER — Forward to port (ignore lifeboats)
    # ══════════════════════════════════════════════════════════════════
    nav.set_cv_phase('deliver', target_types=set())  # no pickups on forward trip
    nav.get_logger().info(f'\n>>> PHASE 1: DELIVER cargo to port ({len(fwd)} waypoints)')
    nav.follow_route(fwd, label='FWD', recalibrate=False)
    nav.get_logger().info('--- Reached target port! ---')
    nav._log_ground_truth('PORT ARRIVED')

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: DROP — Simulate dropping off starting cargo
    # ══════════════════════════════════════════════════════════════════
    nav.get_logger().info('\n>>> PHASE 2: DROP — delivering starting cargo at port')
    time.sleep(1.0)
    nav.get_logger().info('Starting cargo delivered to port ✓')

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: PICKUP CARGO — Back up, then CV-approach target cargo
    # ══════════════════════════════════════════════════════════════════
    cargo_model = f'cargo_{nav.start_pos}'
    nav.set_cv_phase('pickup_cargo', target_types={'cargo'})
    nav.get_logger().info(f'\n>>> PHASE 3: PICKUP target cargo ({cargo_model})')

    # Step back ~0.5m so CV can detect the cargo sitting at the port
    pickup_pose = nav._spin_get_pose()
    if pickup_pose:
        back_y = pickup_pose[1] - 0.5
        nav.goto(pickup_pose[0], back_y, label='[CARGO backup]',
                 gate_dir=None, recalibrate=False)
        # Re-approach port — CV will detect and guide to the cargo
        nav.goto(px, py, label='[CARGO approach]',
                 gate_dir=None, recalibrate=False)

    # Fallback: if CV didn't trigger, force the pickup
    if cargo_model not in nav._cv_picked_up:
        nav.get_logger().info(f'  Force-pickup {cargo_model} (CV missed)')
        nav._delete_gazebo_model(cargo_model)
        nav._cv_picked_up.add(cargo_model)
    nav.get_logger().info('Target cargo picked up ✓')
    nav._log_ground_truth('CARGO PICKED UP')

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: RETURN — Navigate home, pick up lifeboats along the way
    # ══════════════════════════════════════════════════════════════════
    if is_coastal:
        if 'left' in nav.start_pos:
            ret = left_coastal_return()
        else:
            ret = right_coastal_return()
    else:
        ret = open_water_return(sx)

    nav.set_cv_phase('return', target_types={'lifeboat'})
    ret.append((sx, sy))   # final: home deploy zone
    nav.get_logger().info(f'\n>>> PHASE 4: RETURN to deploy ({len(ret)} waypoints)')
    nav.get_logger().info('  CV active: looking for lifeboats to rescue')
    nav.follow_route(ret, label='RET', recalibrate=False)

    # ══════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════
    nav.set_cv_phase('off')
    nav._log_ground_truth('CHALLENGE END')
    nav.get_logger().info('=== Challenge complete! ===')
    if nav._cv_picked_up:
        nav.get_logger().info(f'Objects picked up: {nav._cv_picked_up}')

    nav._stop()
    nav.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
