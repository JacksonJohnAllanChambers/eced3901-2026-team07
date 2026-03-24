#!/usr/bin/env python3
"""
capture_cv_frame.py — Grab one frame from /cv/detection_image and save it,
plus print the current /cv/detections JSON so we can see bbox/aspect values.

Usage:
  ros2 run eced3901 capture_cv_frame.py
"""

import json
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2


class FrameCapture(Node):
    def __init__(self):
        super().__init__('frame_capture')
        self.bridge = CvBridge()
        self.image  = None
        self.dets   = None
        self.create_subscription(Image,  '/cv/detection_image', self._img_cb,  10)
        self.create_subscription(String, '/cv/detections',      self._det_cb,  10)

    def _img_cb(self, msg):
        if self.image is None:
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _det_cb(self, msg):
        if self.dets is None:
            try:
                self.dets = json.loads(msg.data)
            except Exception:
                pass


def main():
    rclpy.init()
    node = FrameCapture()

    print('Waiting for a frame...')
    deadline = time.time() + 10.0
    while time.time() < deadline and (node.image is None or node.dets is None):
        rclpy.spin_once(node, timeout_sec=0.1)

    if node.image is not None:
        path = '/tmp/cv_capture.jpg'
        cv2.imwrite(path, node.image)
        print(f'Image saved: {path}')
    else:
        print('No image received')

    if node.dets is not None:
        print('\n=== Detections ===')
        for d in node.dets:
            bbox = d.get('bbox', [0,0,0,0])
            bx, by, bw, bh = bbox
            print(f"  label={d.get('label')}  conf={d.get('confidence',0):.2f}")
            print(f"  bbox=[x={bx} y={by} w={bw} h={bh}]")
            print(f"  center={d.get('center')}  area={d.get('area',0):.0f}")
            print(f"  aspect_ratio={d.get('aspect_ratio',0):.2f}")
            print(f"  bbox_top={by}  bbox_bottom={by+bh}  bbox_height={bh}")
    else:
        print('No detections received')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
