#!/usr/bin/env python3
"""
CV Web Viewer — MJPEG stream with task/waypoint overlay + pose & minimap.

Subscribes to:
  /cv/detection_image  (sensor_msgs/Image)  — annotated camera frame
  /camera/image_raw    (sensor_msgs/Image)  — fallback raw camera
  /cv/detections       (std_msgs/String)    — detection JSON
  /cv/nav_status       (std_msgs/String)    — current task/phase text
    /cv/cargo_align      (std_msgs/String)    — cargo alignment status JSON
  /odom               (nav_msgs/Odometry)   — odometry for pose
  /imu/imu         (sensor_msgs/Imu)     — IMU for heading

Serves an MJPEG stream at http://<robot_ip>:8080
Open in any browser to see the live camera with overlays.

Usage:
  ros2 run eced3901 cv_viewer.py
  # Then open http://10.0.0.207:8080 in your browser
"""

import glob
import json
import math
import os
import threading
import time
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from cv_bridge import CvBridge
import tf2_ros

HTTP_PORT = 8080
FRAME_W = 960
FRAME_H = 540
JPEG_QUALITY = 70

# ── Mini-map configuration ──
MAP_SIZE       = 180       # pixel size of the minimap square
MAP_MARGIN     = 10        # margin from corner
MAP_BG_COLOR   = (40, 40, 40)
MAP_BORDER     = (100, 100, 100)
MAP_TRAIL_COLOR = (80, 80, 80)
MAP_ROBOT_COLOR = (0, 200, 255)   # cyan-yellow
MAP_HEADING_COLOR = (0, 255, 100) # green heading line
MAP_GRID_COLOR = (60, 60, 60)

# Arena bounds in SLAM frame (auto-adjust from pose data)
ARENA_X_MIN = -1.0
ARENA_X_MAX =  1.5
ARENA_Y_MIN = -0.5
ARENA_Y_MAX =  4.0


class CVViewer(Node):
    def __init__(self):
        super().__init__('cv_viewer')

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self._frame = None
        self._nav_status = ''
        self._detections = []
        self._det_time = 0.0
        self._cargo_align = {}
        self._cargo_align_time = 0.0
        self._frame_count = 0
        self._start_time = time.time()

        # Pose state
        self._pose_x = None
        self._pose_y = None
        self._pose_yaw = None
        self._pose_time = 0.0
        self._trail = deque(maxlen=500)  # position history for minimap trail

        # Auto-adjust map bounds
        self._map_x_min = ARENA_X_MIN
        self._map_x_max = ARENA_X_MAX
        self._map_y_min = ARENA_Y_MIN
        self._map_y_max = ARENA_Y_MAX

        # Map file data for minimap background
        self._maps = []
        self._load_maps()

        # Track nav status timing for trail reset
        self._last_status_time = 0.0

        # Prefer annotated detection image, fall back to raw camera
        self._has_det_image = False
        self.create_subscription(
            Image, '/cv/detection_image', self._det_image_cb, 10)
        self.create_subscription(
            Image, '/camera/image_raw', self._raw_image_cb, 10)
        self.create_subscription(
            String, '/cv/detections', self._detections_cb, 10)
        self.create_subscription(
            String, '/cv/nav_status', self._status_cb, 10)
        self.create_subscription(
            String, '/cv/cargo_align', self._cargo_align_cb, 10)

        # Pose subscribers
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Imu, '/imu/imu', self._imu_cb, 10)

        # TF for map-frame pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f'CV Viewer ready — stream at http://0.0.0.0:{HTTP_PORT}')

    # ── Image callbacks ─────────────────────────────────────────
    def _det_image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.lock:
            self._frame = frame
            self._has_det_image = True
            self._frame_count += 1

    def _raw_image_cb(self, msg):
        with self.lock:
            if self._has_det_image:
                return
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        with self.lock:
            self._frame = frame
            self._frame_count += 1

    def _detections_cb(self, msg):
        try:
            dets = json.loads(msg.data)
        except json.JSONDecodeError:
            dets = []
        with self.lock:
            self._detections = dets
            self._det_time = time.time()

    def _status_cb(self, msg):
        with self.lock:
            now = time.time()
            # Reset trail if status arrives after a gap (new nav command started)
            if self._last_status_time > 0 and now - self._last_status_time > 5.0:
                self._trail.clear()
                self._map_x_min = ARENA_X_MIN
                self._map_x_max = ARENA_X_MAX
                self._map_y_min = ARENA_Y_MIN
                self._map_y_max = ARENA_Y_MAX
                self.get_logger().info('Trail reset — new nav command detected')
            self._nav_status = msg.data
            self._last_status_time = now

    def _cargo_align_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            payload = {}
        with self.lock:
            self._cargo_align = payload
            self._cargo_align_time = time.time()

    # ── Pose callbacks ──────────────────────────────────────────
    def _odom_cb(self, msg):
        # We use TF for position, but odom as fallback
        self._try_update_pose()

    def _imu_cb(self, msg):
        self._try_update_pose()

    def _try_update_pose(self):
        """Try to get map-frame pose from TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.05))
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)

            with self.lock:
                self._pose_x = x
                self._pose_y = y
                self._pose_yaw = yaw
                self._pose_time = time.time()

                # Add to trail (every ~5cm)
                if len(self._trail) == 0:
                    self._trail.append((x, y))
                else:
                    lx, ly = self._trail[-1]
                    if math.hypot(x - lx, y - ly) > 0.05:
                        self._trail.append((x, y))

                # Auto-expand map bounds if robot goes outside
                pad = 0.3
                if x < self._map_x_min + pad:
                    self._map_x_min = x - pad
                if x > self._map_x_max - pad:
                    self._map_x_max = x + pad
                if y < self._map_y_min + pad:
                    self._map_y_min = y - pad
                if y > self._map_y_max - pad:
                    self._map_y_max = y + pad
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            pass

    # ── Map file loading ────────────────────────────────────────
    def _load_maps(self):
        """Load all available PGM map files from the maps directory."""
        maps_dirs = [
            '/home/student/ros2_ws/src/eced3901/maps',
            '/home/student/ros2_ws/install/eced3901/share/eced3901/maps',
        ]
        loaded = set()
        for maps_dir in maps_dirs:
            if not os.path.isdir(maps_dir):
                continue
            for yaml_path in glob.glob(os.path.join(maps_dir, '*.yaml')):
                try:
                    with open(yaml_path) as f:
                        meta = yaml.safe_load(f)
                    pgm_name = meta.get('image', '')
                    if not pgm_name or pgm_name in loaded:
                        continue
                    pgm_path = os.path.join(maps_dir, pgm_name)
                    if not os.path.exists(pgm_path):
                        continue
                    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    loaded.add(pgm_name)
                    resolution = float(meta['resolution'])
                    origin = meta['origin']
                    h, w = img.shape
                    self._maps.append({
                        'image': img,
                        'origin_x': float(origin[0]),
                        'origin_y': float(origin[1]),
                        'resolution': resolution,
                        'width': w,
                        'height': h,
                        'name': pgm_name,
                    })
                    self.get_logger().info(
                        f'Loaded map: {pgm_name} ({w}x{h}, res={resolution})')
                except Exception as e:
                    self.get_logger().warn(f'Failed to load map {yaml_path}: {e}')

    # ── Mini-map rendering ──────────────────────────────────────
    def _draw_minimap(self, frame):
        """Draw a small top-down map in the bottom-left corner."""
        with self.lock:
            px = self._pose_x
            py = self._pose_y
            yaw = self._pose_yaw
            trail = list(self._trail)
            x_min = self._map_x_min
            x_max = self._map_x_max
            y_min = self._map_y_min
            y_max = self._map_y_max

        if px is None:
            return

        ms = MAP_SIZE
        minimap = np.full((ms, ms, 3), MAP_BG_COLOR[0], dtype=np.uint8)
        minimap[:, :, 1] = MAP_BG_COLOR[1]
        minimap[:, :, 2] = MAP_BG_COLOR[2]

        # Coordinate mapping: arena coords -> pixel coords
        # In SLAM frame: +X = north, +Y = west
        #   minimap X-axis (right) = arena +Y
        #   minimap Y-axis (up)    = arena +X (forward/north)
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range < 0.5:
            x_range = 0.5
        if y_range < 0.5:
            y_range = 0.5

        # Use uniform scale with padding
        scale = (ms - 20) / max(x_range, y_range)

        def arena_to_px(ax, ay):
            """Convert arena (x,y) to minimap pixel (col, row)."""
            col = int(10 + (ay - y_min) * scale)
            row = int(ms - 10 - (ax - x_min) * scale)
            return (col, row)

        # ── Render PGM map background ──
        if self._maps:
            mr_arr = np.arange(ms, dtype=np.float32).reshape(-1, 1)
            mc_arr = np.arange(ms, dtype=np.float32).reshape(1, -1)
            # Inverse of arena_to_px: minimap pixel -> world coords
            world_x = (ms - 10 - mr_arr) / scale + x_min
            world_y = (mc_arr - 10) / scale + y_min

            for m in self._maps:
                ox = m['origin_x']
                oy = m['origin_y']
                res = m['resolution']
                mw = m['width']
                mh = m['height']
                pgm = m['image']

                pgm_col = ((world_x - ox) / res).astype(np.int32)
                pgm_row = ((mh - 1) - (world_y - oy) / res).astype(np.int32)

                valid = ((pgm_col >= 0) & (pgm_col < mw) &
                         (pgm_row >= 0) & (pgm_row < mh))

                pgm_col_s = np.clip(pgm_col, 0, mw - 1)
                pgm_row_s = np.clip(pgm_row, 0, mh - 1)
                map_vals = pgm[pgm_row_s, pgm_col_s]

                # Colorize: free (white) -> dark floor,
                #           occupied (black) -> bright walls,
                #           unknown (gray) -> medium
                free = valid & (map_vals > 230)
                occupied = valid & (map_vals < 50)
                unknown = valid & ~free & ~occupied

                minimap[free] = (35, 35, 40)
                minimap[occupied] = (170, 170, 180)
                minimap[unknown] = (50, 45, 45)

        # Draw grid lines at 1m intervals
        for gx in range(int(math.floor(x_min)), int(math.ceil(x_max)) + 1):
            p1 = arena_to_px(gx, y_min)
            p2 = arena_to_px(gx, y_max)
            cv2.line(minimap, p1, p2, MAP_GRID_COLOR, 1)
        for gy in range(int(math.floor(y_min)), int(math.ceil(y_max)) + 1):
            p1 = arena_to_px(x_min, gy)
            p2 = arena_to_px(x_max, gy)
            cv2.line(minimap, p1, p2, MAP_GRID_COLOR, 1)

        # Draw trail
        if len(trail) > 1:
            pts = [arena_to_px(tx, ty) for tx, ty in trail]
            for i in range(1, len(pts)):
                cv2.line(minimap, pts[i - 1], pts[i], MAP_TRAIL_COLOR, 1,
                         cv2.LINE_AA)

        # Draw robot position
        rcol, rrow = arena_to_px(px, py)
        cv2.circle(minimap, (rcol, rrow), 5, MAP_ROBOT_COLOR, -1, cv2.LINE_AA)

        # Draw heading line
        hlen = 18
        dcol = hlen * math.sin(yaw)
        drow = -hlen * math.cos(yaw)
        hend = (int(rcol + dcol), int(rrow + drow))
        cv2.line(minimap, (rcol, rrow), hend, MAP_HEADING_COLOR, 2, cv2.LINE_AA)

        # Border
        cv2.rectangle(minimap, (0, 0), (ms - 1, ms - 1), MAP_BORDER, 1)

        # Label
        cv2.putText(minimap, 'MAP', (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1,
                    cv2.LINE_AA)

        # Composite onto main frame (bottom-left)
        y_off = FRAME_H - ms - MAP_MARGIN
        x_off = MAP_MARGIN

        # Semi-transparent blend
        roi = frame[y_off:y_off + ms, x_off:x_off + ms]
        blended = cv2.addWeighted(minimap, 0.85, roi, 0.15, 0)
        frame[y_off:y_off + ms, x_off:x_off + ms] = blended

    # ── Pose text overlay ───────────────────────────────────────
    def _draw_pose_overlay(self, frame):
        """Draw position and heading text below the top bar."""
        with self.lock:
            px = self._pose_x
            py = self._pose_y
            yaw = self._pose_yaw
            pose_age = time.time() - self._pose_time if self._pose_time > 0 else 999

        if px is None:
            pose_text = 'Pose: waiting for TF...'
            color = (100, 100, 100)
        elif pose_age > 3.0:
            pose_text = f'Pose: STALE ({pose_age:.0f}s ago)'
            color = (80, 80, 200)
        else:
            yaw_deg = math.degrees(yaw)
            # Cardinal direction
            if -22.5 <= yaw_deg < 22.5:
                cardinal = 'N'
            elif 22.5 <= yaw_deg < 67.5:
                cardinal = 'NW'
            elif 67.5 <= yaw_deg < 112.5:
                cardinal = 'W'
            elif 112.5 <= yaw_deg < 157.5:
                cardinal = 'SW'
            elif -67.5 <= yaw_deg < -22.5:
                cardinal = 'NE'
            elif -112.5 <= yaw_deg < -67.5:
                cardinal = 'E'
            elif -157.5 <= yaw_deg < -112.5:
                cardinal = 'SE'
            else:
                cardinal = 'S'
            pose_text = (f'X:{px:+.2f}  Y:{py:+.2f}  '
                         f'Hdg:{yaw_deg:+.0f}\u00b0 ({cardinal})')
            color = (200, 200, 200)

        # Draw below the top bar (y=55)
        # Dark background strip
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 48), (FRAME_W, 74), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, pose_text, (12, 67),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # ── Cargo angle visual overlay ───────────────────────────────
    def _draw_cargo_angle_overlay(self, frame, cargo_align, detections, det_age):
        """Draw centre midline, washer orientation line, and lateral offset tick."""
        # Always draw the vertical midline — cargo centre must land on this
        mid_x = FRAME_W // 2
        cv2.line(frame, (mid_x, 0), (mid_x, FRAME_H), (60, 60, 60), 1, cv2.LINE_AA)
        # Small tick marks every 100px along the midline for scale reference
        for y in range(0, FRAME_H, 100):
            cv2.line(frame, (mid_x - 6, y), (mid_x + 6, y), (80, 80, 80), 1)
        cv2.putText(frame, 'MID', (mid_x + 4, FRAME_H - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1, cv2.LINE_AA)

        if not cargo_align.get('cargo_detected', False):
            return

        yaw_err = cargo_align.get('orientation_error_deg')
        if not isinstance(yaw_err, (int, float)):
            return

        # Find cargo/lifeboat detection center for anchor point
        scale_x = FRAME_W / 1280.0
        scale_y = FRAME_H / 720.0
        cx, cy = FRAME_W // 2, FRAME_H // 2
        if det_age < 2.0:
            for d in detections:
                if d.get('label') in ('cargo', 'lifeboat'):
                    c = d.get('center', [640, 360])
                    cx = int(c[0] * scale_x)
                    cy = int(c[1] * scale_y)
                    break

        # Color by error magnitude
        abs_err = abs(yaw_err)
        if abs_err < 3.0:
            color = (0, 255, 80)    # green  — aligned
        elif abs_err < 8.0:
            color = (0, 200, 255)   # yellow — minor correction
        else:
            color = (0, 80, 255)    # red    — large error

        # Washer orientation line at current angle
        line_len = 90
        angle_rad = math.radians(yaw_err)
        dx = int(line_len * math.cos(angle_rad))
        dy = int(line_len * math.sin(angle_rad))
        cv2.line(frame, (cx - dx, cy - dy), (cx + dx, cy + dy),
                 color, 3, cv2.LINE_AA)

        # Reference horizontal — goal orientation
        cv2.line(frame, (cx - line_len, cy), (cx + line_len, cy),
                 (90, 90, 90), 1, cv2.LINE_AA)

        # Centre dot
        cv2.circle(frame, (cx, cy), 5, color, -1, cv2.LINE_AA)

        # Angle label (large, with shadow)
        angle_text = f'{yaw_err:+.1f}\u00b0'
        tx, ty = cx + line_len + 10, cy + 8
        cv2.putText(frame, angle_text, (tx + 1, ty + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, angle_text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Lateral offset tick
        offset_x = cargo_align.get('offset_x')
        if isinstance(offset_x, (int, float)):
            off_px = int(offset_x * scale_x)
            frame_cx = FRAME_W // 2
            tick_x = frame_cx + off_px
            # Reference centre mark
            cv2.line(frame, (frame_cx, cy - 18), (frame_cx, cy + 18),
                     (70, 70, 70), 1, cv2.LINE_AA)
            # Offset tick
            off_color = (255, 200, 0)
            cv2.line(frame, (tick_x, cy - 22), (tick_x, cy + 22),
                     off_color, 2, cv2.LINE_AA)
            off_text = f'dx {offset_x:+.0f}px'
            cv2.putText(frame, off_text, (tick_x - 38, cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, off_color, 1, cv2.LINE_AA)

    # ── Main render ─────────────────────────────────────────────
    def get_jpeg(self):
        """Render the current frame with all overlays and return JPEG bytes."""
        with self.lock:
            frame = self._frame
            status = self._nav_status
            detections = list(self._detections)
            cargo_align = dict(self._cargo_align)
            det_age = (time.time() - self._det_time
                       if self._det_time > 0 else 999)
            cargo_age = (time.time() - self._cargo_align_time
                         if self._cargo_align_time > 0 else 999)
            fc = self._frame_count

        if frame is None:
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            cv2.putText(frame, 'Waiting for camera...', (60, FRAME_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
        else:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        # ── Top bar: nav status ──
        bar_h = 48
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (FRAME_W, bar_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        if status:
            cv2.putText(frame, status, (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 200), 2,
                        cv2.LINE_AA)
        else:
            cv2.putText(frame, 'No nav status', (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1,
                        cv2.LINE_AA)

        # ── Pose bar (below top bar) ──
        self._draw_pose_overlay(frame)

        # ── Cargo align labels (below pose bar) ──
        if cargo_age < 2.0 and cargo_align:
            overlay_align = frame.copy()
            cv2.rectangle(overlay_align, (0, 76), (FRAME_W, 106), (20, 20, 20), -1)
            cv2.addWeighted(overlay_align, 0.7, frame, 0.3, 0, frame)

            c_state = cargo_align.get('state', '?')
            c_ax = cargo_align.get('aligned_x', False)
            c_at = cargo_align.get('aligned_theta', False)
            c_ay = cargo_align.get('aligned_y', False)
            c_theta = cargo_align.get('orientation_error_deg', None)
            c_src = cargo_align.get('orientation_source', '-')
            c_off = cargo_align.get('offset_x', None)
            c_stable = cargo_align.get('ready_stable_count', 0)
            c_need = cargo_align.get('ready_stable_required', 0)

            flags = f"X:{'Y' if c_ax else 'n'} T:{'Y' if c_at else 'n'} Y:{'Y' if c_ay else 'n'}"
            theta_text = f"th={c_theta:+.1f}°" if isinstance(c_theta, (int, float)) else "th=--"
            off_text = f"dx={c_off:+.0f}px" if isinstance(c_off, (int, float)) else "dx=--"
            align_text = (
                f"CargoAlign {c_state}  {flags}  {theta_text} ({c_src})  {off_text}  stable {c_stable}/{c_need}"
            )
            color = (0, 255, 120) if (c_ax and c_at and c_ay) else (80, 220, 255)
            cv2.putText(frame, align_text, (12, 97),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        # ── Cargo angle visual overlay ──
        self._draw_cargo_angle_overlay(frame, cargo_align, detections, det_age)

        # ── Bottom bar: detection summary ──
        bot_y = FRAME_H - 36
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, bot_y - 4), (FRAME_W, FRAME_H),
                      (30, 30, 30), -1)
        cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

        if det_age < 2.0 and detections:
            labels = {}
            for d in detections:
                lbl = d.get('label', '?')
                conf = d.get('confidence', 0)
                area = d.get('area', 0)
                labels[lbl] = f'{lbl} c={conf:.2f} a={area:.0f}'
            det_text = '  |  '.join(labels.values())
            color = (0, 255, 100)
        elif det_age < 2.0:
            det_text = 'No objects detected'
            color = (80, 80, 200)
        else:
            det_text = ('Detections stale' if self._det_time > 0
                        else 'Waiting for detections...')
            color = (100, 100, 100)

        cv2.putText(frame, det_text, (12, FRAME_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

        # ── FPS counter (top-right) ──
        elapsed = time.time() - self._start_time
        fps = fc / elapsed if elapsed > 0 else 0
        fps_text = f'{fps:.0f} fps'
        cv2.putText(frame, fps_text, (FRAME_W - 110, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1,
                    cv2.LINE_AA)

        # ── Mini-map (bottom-left) ──
        self._draw_minimap(frame)

        # Encode to JPEG
        ok, buf = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return b''
        return buf.tobytes()


# ── MJPEG HTTP handler ──────────────────────────────────────────
_viewer_node = None


class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=--frame')
            self.send_header('Cache-Control',
                             'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            try:
                while rclpy.ok():
                    jpeg = _viewer_node.get_jpeg()
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(
                        f'Content-Length: {len(jpeg)}\r\n'.encode())
                    self.wfile.write(b'\r\n')
                    self.wfile.write(jpeg)
                    self.wfile.write(b'\r\n')
                    time.sleep(1.0 / 15)  # ~15 fps stream
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == '/snapshot':
            jpeg = _viewer_node.get_jpeg()
            self.send_response(200)
            self.send_header('Content-Type', 'image/jpeg')
            self.send_header('Content-Length', str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
        else:
            html = f"""<!DOCTYPE html>
<html><head>
<title>DaliBot CV Viewer</title>
<style>
body {{ margin:0; background:#111; display:flex; justify-content:center;
       align-items:center; height:100vh; font-family:monospace; }}
img {{ max-width:100%; max-height:100vh; }}
</style>
</head><body>
<img src="/stream" alt="CV Stream" />
</body></html>"""
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())

    def log_message(self, fmt, *args):
        pass


def main():
    global _viewer_node
    rclpy.init()
    _viewer_node = CVViewer()

    server = HTTPServer(('0.0.0.0', HTTP_PORT), MJPEGHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    _viewer_node.get_logger().info(f'MJPEG server started on port {HTTP_PORT}')

    try:
        rclpy.spin(_viewer_node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        _viewer_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
