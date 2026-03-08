#!/usr/bin/env python3
"""
CV Web Viewer — MJPEG stream with task/waypoint overlay + pose & minimap.

Subscribes to:
  /cv/detection_image  (sensor_msgs/Image)  — annotated camera frame
  /camera/image_raw    (sensor_msgs/Image)  — fallback raw camera
  /cv/detections       (std_msgs/String)    — detection JSON
  /cv/nav_status       (std_msgs/String)    — current task/phase text
  /odom               (nav_msgs/Odometry)   — odometry for pose
  /bno055/imu         (sensor_msgs/Imu)     — IMU for heading

Serves an MJPEG stream at http://<robot_ip>:8080
Open in any browser to see the live camera with overlays.

Usage:
  ros2 run eced3901 cv_viewer.py
  # Then open http://10.0.0.207:8080 in your browser
"""

import json
import math
import threading
import time
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler

import cv2
import numpy as np

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

        # Pose subscribers
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Imu, '/bno055/imu', self._imu_cb, 10)

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
            self._nav_status = msg.data

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
        # On minimap: we show X horizontal, Y vertical (Y up = north on map)
        # Actually let's orient it intuitively:
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
            # Map arena Y to pixel X (left=y_min, right=y_max)
            col = int(10 + (ay - y_min) * scale)
            # Map arena X to pixel Y (bottom=x_min, top=x_max) — flip Y
            row = int(ms - 10 - (ax - x_min) * scale)
            return (col, row)

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
        # In SLAM frame: yaw=0 is +X, yaw=pi/2 is +Y
        # On minimap: +X maps to up (negative row), +Y maps to right (positive col)
        hlen = 18
        dcol = hlen * math.sin(yaw)   # +Y component
        drow = -hlen * math.cos(yaw)  # +X component (flipped for screen)
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

    # ── Main render ─────────────────────────────────────────────
    def get_jpeg(self):
        """Render the current frame with all overlays and return JPEG bytes."""
        with self.lock:
            frame = self._frame
            status = self._nav_status
            detections = list(self._detections)
            det_age = (time.time() - self._det_time
                       if self._det_time > 0 else 999)
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
