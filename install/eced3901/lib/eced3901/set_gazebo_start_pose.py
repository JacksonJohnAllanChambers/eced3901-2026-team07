#! /usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState


FT = 0.3048
NORTH = math.pi / 2.0

START_POSES = {
    'left_coastal': (3 * FT, 1 * FT, NORTH),
    'left_open': (5 * FT, 1 * FT, NORTH),
    'right_open': (9 * FT, 1 * FT, NORTH),
    'right_coastal': (11 * FT, 1 * FT, NORTH),
}


class GazeboStartPoseSetter(Node):
    def __init__(self):
        super().__init__('gazebo_start_pose_setter')

        self.declare_parameter('start', 'left_coastal')
        self.start = self.get_parameter('start').get_parameter_value().string_value
        if self.start not in START_POSES:
            self.get_logger().warn(
                f"Unknown start '{self.start}', defaulting to left_coastal")
            self.start = 'left_coastal'

        self.model_names = []
        self.model_candidates = [
            'eced3901bot_description',
            'eced3901bot',
            'eced3901bot_description_0',
            'eced3901bot_0',
        ]
        self.done = False
        self.request_in_flight = False
        self.attempts = 0
        self.max_attempts = 240

        self.create_subscription(ModelStates, '/gazebo/model_states', self._model_states_cb, 10)
        self.create_subscription(ModelStates, '/model_states', self._model_states_cb, 10)
        self.client_gazebo = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.client_root = self.create_client(SetEntityState, '/set_entity_state')
        self.timer = self.create_timer(0.25, self._tick)

    def _model_states_cb(self, msg):
        self.model_names = list(msg.name)
        for name in self.model_names:
            if 'eced3901bot' in name and name not in self.model_candidates:
                self.model_candidates.append(name)

    def _build_request(self, model_name: str):
        x, y, yaw = START_POSES[self.start]
        req = SetEntityState.Request()
        req.state.name = model_name
        req.state.reference_frame = 'world'
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = 0.0
        req.state.pose.orientation.x = 0.0
        req.state.pose.orientation.y = 0.0
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state.twist.linear.x = 0.0
        req.state.twist.linear.y = 0.0
        req.state.twist.linear.z = 0.0
        req.state.twist.angular.x = 0.0
        req.state.twist.angular.y = 0.0
        req.state.twist.angular.z = 0.0
        return req

    def _tick(self):
        if self.done:
            return
        if self.request_in_flight:
            return
        self.attempts += 1
        if self.attempts > self.max_attempts:
            self.get_logger().error('Timed out waiting to set Gazebo start pose')
            self.done = True
            return

        client = None
        if self.client_gazebo.wait_for_service(timeout_sec=0.0):
            client = self.client_gazebo
        elif self.client_root.wait_for_service(timeout_sec=0.0):
            client = self.client_root
        if client is None:
            return

        tried = set()
        for model_name in self.model_candidates:
            if model_name in tried:
                continue
            tried.add(model_name)
            req = self._build_request(model_name)
            self.request_in_flight = True
            future = client.call_async(req)
            future.add_done_callback(
                lambda fut, chosen=model_name: self._on_response(fut, chosen))
            return

    def _on_response(self, future, model_name):
        self.request_in_flight = False
        try:
            resp = future.result()
        except Exception as exc:
            self.get_logger().error(f'SetEntityState call failed: {exc}')
            return

        if resp.success:
            x, y, _ = START_POSES[self.start]
            self.get_logger().info(
                f"Set '{model_name}' to start={self.start} at ({x:.3f}, {y:.3f})")
            self.done = True
            self.timer.cancel()
            return

        if resp.status_message and 'exist' not in resp.status_message.lower():
            self.get_logger().warn(f'SetEntityState rejected for {model_name}: {resp.status_message}')


def main(args=None):
    rclpy.init(args=args)
    node = GazeboStartPoseSetter()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
