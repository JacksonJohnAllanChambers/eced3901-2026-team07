#!/usr/bin/env python3
"""
CV Issue Profiler — Capture and label problematic detection frames.
===================================================================
Run alongside detection_node to diagnose CV failures.

Subscribes to:
  /camera/image_raw    — raw camera frame
  /cv/detections       — JSON detection results
  /cv/detection_image  — annotated debug frame

Shows a live view with detection overlays.  Press a key to capture
the current frame with a diagnostic label:

  1  —  boat in frame but NOT detected
  2  —  boat detected as cargo
  3  —  cargo detected as boat (lifeboat)
  4  —  cargo in frame but NOT detected
  5  —  floor / shadow detected as lifeboat (false positive)
  6  —  correct detection (positive sample)
  7  —  custom label (enter in terminal)
  s  —  save raw frame only (no label)
  q  —  quit

Saves to ~/cv_profiler_captures/:
  {timestamp}_{label}.jpg           — raw frame
  {timestamp}_{label}_annotated.jpg — annotated frame
  {timestamp}_{label}.json          — metadata + detections

Usage (on robot):
  ros2 run eced3901 cv_profiler
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import numpy as np

# ── Label definitions ─────────────────────────────────────────────
LABELS = {
    ord('1'): 'boat_not_detected',
    ord('2'): 'boat_as_cargo',
    ord('3'): 'cargo_as_boat',
    ord('4'): 'cargo_not_detected',
    ord('5'): 'floor_as_lifeboat',
    ord('6'): 'correct_detection',
    ord('7'): 'custom',
}

LABEL_DESCRIPTIONS = {
    'boat_not_detected':  'Lifeboat visible but not detected',
    'boat_as_cargo':      'Lifeboat misclassified as cargo',
    'cargo_as_boat':      'Cargo misclassified as lifeboat',
    'cargo_not_detected': 'Cargo visible but not detected',
    'floor_as_lifeboat':  'Floor/shadow false positive as lifeboat',
    'correct_detection':  'Detection is correct (positive sample)',
    'custom':             'Custom label',
}


class CVProfiler(Node):
    def __init__(self):
        super().__init__('cv_profiler')

        self.bridge = CvBridge()
        self.raw_frame = None
        self.annotated_frame = None
        self.detections = []
        self.det_json_str = '[]'
        self.capture_count = 0

        # Save directory
        self.save_dir = Path.home() / 'cv_profiler_captures'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f'Saving captures to {self.save_dir}')

        # Subscribers
        self.create_subscription(
            Image, '/camera/image_raw', self._raw_cb, 10)
        self.create_subscription(
            String, '/cv/detections', self._det_cb, 10)
        self.create_subscription(
            Image, '/cv/detection_image', self._ann_cb, 10)

        # Timer for display loop (30 Hz)
        self.create_timer(1.0 / 30.0, self._display_loop)

        self.get_logger().info('CV Profiler ready — press keys in the window to capture')
        self._print_help()

    def _print_help(self):
        print('\n╔══════════════════════════════════════════╗')
        print('║        CV Issue Profiler Controls        ║')
        print('╠══════════════════════════════════════════╣')
        print('║  1  Boat in frame but NOT detected       ║')
        print('║  2  Boat detected as cargo               ║')
        print('║  3  Cargo detected as boat               ║')
        print('║  4  Cargo NOT detected                   ║')
        print('║  5  Floor/shadow false positive           ║')
        print('║  6  Correct detection (positive sample)   ║')
        print('║  7  Custom label (type in terminal)       ║')
        print('║  s  Save raw frame (no label)             ║')
        print('║  q  Quit                                  ║')
        print('╚══════════════════════════════════════════╝\n')

    # ── Callbacks ─────────────────────────────────────────────────
    def _raw_cb(self, msg: Image):
        self.raw_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def _det_cb(self, msg: String):
        self.det_json_str = msg.data
        try:
            self.detections = json.loads(msg.data)
        except json.JSONDecodeError:
            self.detections = []

    def _ann_cb(self, msg: Image):
        self.annotated_frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    # ── Display loop ──────────────────────────────────────────────
    def _display_loop(self):
        if self.raw_frame is None:
            return

        # Build display: annotated if available, else raw with overlay
        if self.annotated_frame is not None:
            display = self.annotated_frame.copy()
        else:
            display = self.raw_frame.copy()

        # HUD overlay
        h, w = display.shape[:2]

        # Detection summary
        lifeboats = [d for d in self.detections if d.get('label') == 'lifeboat']
        cargos = [d for d in self.detections if d.get('label') == 'cargo']
        det_text = f'L:{len(lifeboats)} C:{len(cargos)}'
        cv2.putText(display, det_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show detection details
        y_off = 50
        for d in self.detections:
            label = d.get('label', '?')
            conf = d.get('confidence', 0)
            area = d.get('area', 0)
            cr = d.get('local_contrast', 0)
            info = f'{label} conf={conf:.2f} area={area:.0f} cr={cr:.2f}'
            color = (0, 255, 0) if label == 'lifeboat' else (255, 165, 0)
            cv2.putText(display, info, (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_off += 20

        # Capture count
        cv2.putText(display, f'Captures: {self.capture_count}', (w - 180, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Key hint
        cv2.putText(display, '1-7:label s:save q:quit', (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow('CV Profiler', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.get_logger().info(
                f'Quitting — {self.capture_count} frames captured')
            cv2.destroyAllWindows()
            raise SystemExit(0)

        elif key == ord('s'):
            self._save_capture('unlabeled')

        elif key in LABELS:
            label = LABELS[key]
            if label == 'custom':
                print('Enter custom label: ', end='', flush=True)
                custom = input().strip()
                if custom:
                    # Sanitize: only keep alphanumeric, underscores, hyphens
                    label = ''.join(
                        c if c.isalnum() or c in '_-' else '_'
                        for c in custom
                    )
                else:
                    self.get_logger().warn('Empty label — skipping capture')
                    return
            self._save_capture(label)

    # ── Save capture ──────────────────────────────────────────────
    def _save_capture(self, label: str):
        if self.raw_frame is None:
            self.get_logger().warn('No frame to save')
            return

        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        prefix = f'{ts}_{label}'

        # Save raw frame
        raw_path = self.save_dir / f'{prefix}.jpg'
        cv2.imwrite(str(raw_path), self.raw_frame)

        # Save annotated frame if available
        if self.annotated_frame is not None:
            ann_path = self.save_dir / f'{prefix}_annotated.jpg'
            cv2.imwrite(str(ann_path), self.annotated_frame)

        # Save metadata
        metadata = {
            'timestamp': ts,
            'label': label,
            'description': LABEL_DESCRIPTIONS.get(label, label),
            'detections': self.detections,
            'detection_count': {
                'lifeboat': len([d for d in self.detections
                                 if d.get('label') == 'lifeboat']),
                'cargo': len([d for d in self.detections
                              if d.get('label') == 'cargo']),
            },
            'frame_shape': list(self.raw_frame.shape),
        }
        meta_path = self.save_dir / f'{prefix}.json'
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.capture_count += 1
        desc = LABEL_DESCRIPTIONS.get(label, label)
        self.get_logger().info(
            f'[{self.capture_count}] Captured: {label} — {desc}')

        # Flash green border on display as feedback
        if self.raw_frame is not None:
            flash = self.raw_frame.copy()
            cv2.rectangle(flash, (0, 0),
                          (flash.shape[1] - 1, flash.shape[0] - 1),
                          (0, 255, 0), 8)
            cv2.putText(flash, f'SAVED: {label}',
                        (flash.shape[1] // 2 - 120, flash.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.imshow('CV Profiler', flash)
            cv2.waitKey(300)


def main(args=None):
    rclpy.init(args=args)
    node = CVProfiler()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
