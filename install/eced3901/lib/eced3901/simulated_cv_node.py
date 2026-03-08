#!/usr/bin/env python3
"""
Simulated Computer Vision Node
===============================
Simulates camera-based object detection by checking proximity to known
Gazebo model positions.  No actual camera is needed.

Subscribes to /gazebo/model_states for real-time object positions
(no slow synchronous service calls).

Detectable objects:
  - cargo_*    : detected at 0.2–2.0 m, in front ±60°
  - lifeboat_* : detected at 0.2–2.0 m, in front ±60°

Publishes:
  /cv/detections        – std_msgs/String (JSON per detection)
  /cv/detection_marker  – visualization_msgs/Marker (RViz sphere)

Runs at 10 Hz with simulated noise (bearing jitter, distance jitter,
5% frame-drop probability).
"""

import json
import math
import random

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from gazebo_msgs.msg import ModelStates
import tf2_ros

# Object type → detection range limits
OBJECT_TYPES = {
    'cargo':    {'min_range': 0.2, 'max_range': 2.0},
    'lifeboat': {'min_range': 0.2, 'max_range': 2.0},
}

# Field of view half-angle (±60°)
FOV_HALF = math.pi / 3.0

# Noise parameters
BEARING_JITTER  = 0.05   # rad
DISTANCE_JITTER = 0.03   # m
DROP_RATE       = 0.05   # probability of missed frame per object per tick


class SimulatedCVNode(Node):
    def __init__(self):
        super().__init__('simulated_cv')

        # Parameters
        self.declare_parameter('detection_range', 2.0)
        self.declare_parameter('rate', 10.0)

        self.det_range = self.get_parameter('detection_range').value
        rate_hz = self.get_parameter('rate').value

        # Publishers
        self.det_pub = self.create_publisher(String, '/cv/detections', 10)
        self.marker_pub = self.create_publisher(Marker, '/cv/detection_marker', 10)

        # TF for robot pose
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to Gazebo model states (fast, no service calls)
        self._model_positions = {}  # name → (x, y)
        self.create_subscription(
            ModelStates, '/gazebo/model_states', self._model_states_cb, 10)

        # Timer
        self.create_timer(1.0 / rate_hz, self._tick)
        self.get_logger().info(
            f'Simulated CV node started (range={self.det_range}m, rate={rate_hz}Hz)')

    # ── Gazebo model states callback ──────────────────────────────────

    def _model_states_cb(self, msg):
        """Cache positions of cargo_* and lifeboat_* models."""
        for name, pose in zip(msg.name, msg.pose):
            if name.startswith('cargo_') or name.startswith('lifeboat_'):
                self._model_positions[name] = (pose.position.x, pose.position.y)

    # ── Robot pose ────────────────────────────────────────────────────

    def _get_robot_pose(self):
        """Get robot (x, y, yaw) from TF map→base_footprint."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_footprint', rclpy.time.Time(),
                timeout=Duration(seconds=0.1))
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

    # ── Main loop ─────────────────────────────────────────────────────

    def _tick(self):
        if not self._model_positions:
            return

        robot = self._get_robot_pose()
        if robot is None:
            return
        rx, ry, ryaw = robot

        for name, (ox, oy) in list(self._model_positions.items()):
            # Determine object type from name prefix
            if name.startswith('cargo_'):
                obj_type = 'cargo'
            elif name.startswith('lifeboat_'):
                obj_type = 'lifeboat'
            else:
                continue

            dx = ox - rx
            dy = oy - ry
            dist = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx) - ryaw
            # Normalize bearing to [-pi, pi]
            while bearing > math.pi:
                bearing -= 2.0 * math.pi
            while bearing < -math.pi:
                bearing += 2.0 * math.pi

            # Check detection criteria
            limits = OBJECT_TYPES[obj_type]
            if dist < limits['min_range'] or dist > min(limits['max_range'], self.det_range):
                continue
            if abs(bearing) > FOV_HALF:
                continue

            # Simulate frame drop
            if random.random() < DROP_RATE:
                continue

            # Add noise
            noisy_dist = dist + random.uniform(-DISTANCE_JITTER, DISTANCE_JITTER)
            noisy_bearing = bearing + random.uniform(-BEARING_JITTER, BEARING_JITTER)
            confidence = max(0.0, min(1.0,
                1.0 - 0.3 * (dist / limits['max_range'])
                + random.uniform(-0.05, 0.05)))

            # Publish detection JSON
            det = {
                'type': obj_type,
                'distance': round(noisy_dist, 3),
                'bearing': round(noisy_bearing, 4),
                'confidence': round(confidence, 3),
                'model_name': name,
            }
            msg = String()
            msg.data = json.dumps(det)
            self.det_pub.publish(msg)

            # Publish RViz marker
            self._publish_marker(name, ox, oy, obj_type)

    def _publish_marker(self, name, x, y, obj_type):
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'cv_detections'
        m.id = hash(name) % 2147483647
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.15
        m.pose.orientation.w = 1.0
        m.scale.x = 0.12
        m.scale.y = 0.12
        m.scale.z = 0.12
        if obj_type == 'cargo':
            m.color.r = 0.4
            m.color.g = 0.4
            m.color.b = 0.4
            m.color.a = 0.9
        else:  # lifeboat
            m.color.r = 1.0
            m.color.g = 0.5
            m.color.b = 0.0
            m.color.a = 0.9
        m.lifetime.sec = 0
        m.lifetime.nanosec = 300000000  # 0.3s
        self.marker_pub.publish(m)


def main():
    rclpy.init()
    node = SimulatedCVNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
