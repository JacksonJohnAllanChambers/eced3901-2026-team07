#!/usr/bin/env python3
"""
Spawn or delete course objects (cargo blocks & lifeboats) in Gazebo.

Cargo:     One per path, at the centre of each port zone, random 90° rotation.
Lifeboats: One per path, random position inside the lane but OUTSIDE port zones.

Usage:
  ros2 run eced3901 spawn_course_objects.py --spawn-all     # spawn 4 cargo + 4 lifeboats
  ros2 run eced3901 spawn_course_objects.py --delete-all    # remove everything
  ros2 run eced3901 spawn_course_objects.py --type cargo --path left_coastal           # single cargo
  ros2 run eced3901 spawn_course_objects.py --type lifeboat --path right_open --delete # single delete
"""
import argparse
import math
import random
import sys

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

# ── Arena geometry ─────────────────────────────────────────────────────────
# Port zone centres (green 0.610×0.610 markers at top of arena)
PORT_CENTRES = {
    'left_coastal':  (0.914, 3.962),
    'left_open':     (1.524, 3.962),
    'right_open':    (2.743, 3.962),
    'right_coastal': (3.353, 3.962),
}

# Lifeboat placement zones per path.
# x_min/x_max = lane inner-wall bounds with ~0.12 m buffer.
# y_bands = navigable y-ranges between zigzag walls, below port zones.
# Coastal paths have 3 zigzag walls at y ≈ 1.219, 2.134, 3.048 (each 0.15 thick).
# Open paths have no zigzag walls.
LIFEBOAT_ZONES = {
    'left_coastal': {
        'x_min': 0.20, 'x_max': 1.05,
        'y_bands': [
            (0.20, 1.10),   # deploy zone area → below wall 1
            (1.35, 2.00),   # between wall 1 and wall 2
            (2.25, 2.90),   # between wall 2 and wall 3
            (3.17, 3.60),   # above wall 3, below port zone
        ],
    },
    'left_open': {
        'x_min': 1.40, 'x_max': 2.05,
        'y_bands': [(0.20, 3.60)],
    },
    'right_open': {
        'x_min': 2.20, 'x_max': 2.87,
        'y_bands': [(0.20, 3.60)],
    },
    'right_coastal': {
        'x_min': 3.22, 'x_max': 4.07,
        'y_bands': [
            (0.20, 1.10),
            (1.35, 2.00),
            (2.25, 2.90),
            (3.17, 3.60),
        ],
    },
}

ALL_PATHS = ['left_coastal', 'left_open', 'right_open', 'right_coastal']

# Heights (half-height so bottom sits on ground)
CARGO_Z    = 0.0381   # 3-inch block half-height
LIFEBOAT_Z = 0.0254   # 2-inch block half-height

# ── SDF templates ──────────────────────────────────────────────────────────
CARGO_SDF = """\
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="block">
        <geometry><box><size>0.1524 0.0762 0.0762</size></box></geometry>
        <material><ambient>0.3 0.3 0.3 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material>
      </visual>
      <visual name="washer1">
        <pose>-0.035 0 0.0381 0 0 0</pose>
        <geometry><cylinder><radius>0.0127</radius><length>0.003</length></cylinder></geometry>
        <material><ambient>0.75 0.75 0.75 1</ambient><diffuse>0.75 0.75 0.75 1</diffuse></material>
      </visual>
      <visual name="washer2">
        <pose>0.035 0 0.0381 0 0 0</pose>
        <geometry><cylinder><radius>0.0127</radius><length>0.003</length></cylinder></geometry>
        <material><ambient>0.75 0.75 0.75 1</ambient><diffuse>0.75 0.75 0.75 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""

LIFEBOAT_SDF = """\
<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="hull">
        <geometry><box><size>0.1016 0.0508 0.0508</size></box></geometry>
        <material><ambient>1.0 0.55 0.0 1</ambient><diffuse>1.0 0.55 0.0 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>"""


# ── Helpers ────────────────────────────────────────────────────────────────
def _model_name(obj_type, path):
    return f'{obj_type}_{path}'


def _random_cargo_pose(path):
    """Centre of port zone, random 90° rotation."""
    cx, cy = PORT_CENTRES[path]
    yaw = random.choice([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    return cx, cy, CARGO_Z, yaw


def _random_lifeboat_pose(path):
    """Random position inside the lane, outside port zones."""
    zone = LIFEBOAT_ZONES[path]
    band = random.choice(zone['y_bands'])
    x = random.uniform(zone['x_min'], zone['x_max'])
    y = random.uniform(band[0], band[1])
    yaw = random.uniform(0, 2 * math.pi)
    return x, y, LIFEBOAT_Z, yaw


# ── ROS node ───────────────────────────────────────────────────────────────
class ObjectSpawner(Node):
    def __init__(self):
        super().__init__('object_spawner')
        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_cli = self.create_client(DeleteEntity, '/delete_entity')

    def _wait(self, client, timeout=10.0):
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f'Service {client.srv_name} not available')
            return False
        return True

    def spawn(self, name, sdf_template, x, y, z, yaw):
        if not self._wait(self.spawn_cli):
            return False
        sdf = sdf_template.format(name=name)
        req = SpawnEntity.Request()
        req.name = name
        req.xml = sdf
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        req.initial_pose.orientation.z = math.sin(yaw / 2.0)
        req.initial_pose.orientation.w = math.cos(yaw / 2.0)

        future = self.spawn_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is not None and future.result().success:
            self.get_logger().info(f'Spawned {name} at ({x:.3f}, {y:.3f}, yaw={yaw:.2f})')
            return True
        msg = future.result().status_message if future.result() else 'timeout'
        self.get_logger().error(f'Spawn {name} failed: {msg}')
        return False

    def delete(self, name):
        if not self._wait(self.delete_cli):
            return False
        req = DeleteEntity.Request()
        req.name = name
        future = self.delete_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if future.result() is not None and future.result().success:
            self.get_logger().info(f'Deleted {name}')
            return True
        msg = future.result().status_message if future.result() else 'timeout'
        self.get_logger().error(f'Delete {name} failed: {msg}')
        return False

    # ── batch operations ───────────────────────────────────────────────
    def spawn_all(self):
        """Spawn 4 cargo blocks + 4 lifeboats with randomised placement."""
        for path in ALL_PATHS:
            # Cargo at port centre, random 90° rotation
            x, y, z, yaw = _random_cargo_pose(path)
            self.spawn(_model_name('cargo', path), CARGO_SDF, x, y, z, yaw)

            # Lifeboat at random position in lane (outside port zones)
            x, y, z, yaw = _random_lifeboat_pose(path)
            self.spawn(_model_name('lifeboat', path), LIFEBOAT_SDF, x, y, z, yaw)

    def delete_all(self):
        """Delete all 4 cargo blocks + 4 lifeboats."""
        for path in ALL_PATHS:
            self.delete(_model_name('cargo', path))
            self.delete(_model_name('lifeboat', path))


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Spawn/delete course objects in Gazebo')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--spawn-all', action='store_true',
                       help='Spawn all 4 cargo + 4 lifeboats (randomised)')
    group.add_argument('--delete-all', action='store_true',
                       help='Delete all cargo + lifeboats')
    group.add_argument('--type', choices=['cargo', 'lifeboat'],
                       help='Spawn/delete a single object type')

    parser.add_argument('--path', choices=ALL_PATHS,
                        help='Which path (required for single spawn/delete)')
    parser.add_argument('--delete', action='store_true',
                        help='Delete instead of spawn (with --type)')
    parser.add_argument('--x', type=float, default=None, help='Override X position')
    parser.add_argument('--y', type=float, default=None, help='Override Y position')
    parser.add_argument('--yaw', type=float, default=None, help='Override yaw (rad)')
    args = parser.parse_args()

    rclpy.init()
    node = ObjectSpawner()

    try:
        if args.spawn_all:
            node.spawn_all()
        elif args.delete_all:
            node.delete_all()
        else:
            # Single object mode
            if not args.path:
                parser.error('--path is required with --type')
            name = _model_name(args.type, args.path)
            if args.delete:
                node.delete(name)
            else:
                if args.type == 'cargo':
                    x, y, z, yaw = _random_cargo_pose(args.path)
                else:
                    x, y, z, yaw = _random_lifeboat_pose(args.path)
                # Allow CLI overrides
                if args.x is not None:
                    x = args.x
                if args.y is not None:
                    y = args.y
                if args.yaw is not None:
                    yaw = args.yaw
                sdf = CARGO_SDF if args.type == 'cargo' else LIFEBOAT_SDF
                node.spawn(name, sdf, x, y, z, yaw)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
