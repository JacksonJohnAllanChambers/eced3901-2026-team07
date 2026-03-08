#!/usr/bin/env python3
"""
Nav Stack Diagnostic Logger
============================
Prints a snapshot of all key nav data every second so issues can be
copy-pasted for debugging.

Usage (with arena_gazebo + arena_amcl running):
  ros2 run eced3901 nav_diag.py

Optionally drive forward briefly to test odom tracking:
  ros2 run eced3901 nav_diag.py --ros-args -p drive_test:=true
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
import tf2_ros


def quat_to_yaw(q):
    """Extract yaw from quaternion (x,y,z,w)."""
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class NavDiag(Node):
    def __init__(self):
        super().__init__('nav_diag')
        self.declare_parameter('drive_test', False)
        self.do_drive = self.get_parameter('drive_test').get_parameter_value().bool_value

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribers – store latest message
        self.last_odom = None
        self.last_imu = None
        self.last_scan = None
        self.odom_count = 0
        self.imu_count = 0
        self.scan_count = 0

        # Try both /odom and /wheel/odometry
        self.odom_topic = None
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Odometry, '/wheel/odometry', self._wheel_odom_cb, 10)
        self.create_subscription(Imu, '/imu/data', self._imu_cb, 10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.last_wheel_odom = None
        self.wheel_odom_count = 0

        # cmd_vel for drive test
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

    def _odom_cb(self, msg):
        self.last_odom = msg
        self.odom_count += 1
        self.odom_topic = '/odom'

    def _wheel_odom_cb(self, msg):
        self.last_wheel_odom = msg
        self.wheel_odom_count += 1

    def _imu_cb(self, msg):
        self.last_imu = msg
        self.imu_count += 1

    def _scan_cb(self, msg):
        self.last_scan = msg
        self.scan_count += 1

    def _lookup_tf(self, parent, child):
        try:
            t = self.tf_buffer.lookup_transform(parent, child, rclpy.time.Time(),
                                                 timeout=Duration(seconds=0.3))
            x = t.transform.translation.x
            y = t.transform.translation.y
            q = t.transform.rotation
            yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            return (x, y, yaw)
        except Exception as e:
            return None

    def print_snapshot(self, label=""):
        print(f"\n{'='*65}")
        print(f"  NAV DIAGNOSTIC SNAPSHOT  {label}")
        print(f"{'='*65}")

        # TF transforms
        for parent, child in [('map', 'odom'), ('odom', 'base_footprint'),
                               ('map', 'base_footprint'), ('base_footprint', 'base_link'),
                               ('base_link', 'lidar_link')]:
            tf = self._lookup_tf(parent, child)
            if tf:
                print(f"  TF {parent:>15s} → {child:<18s}  x={tf[0]:+7.3f}  y={tf[1]:+7.3f}  yaw={math.degrees(tf[2]):+7.1f}°")
            else:
                print(f"  TF {parent:>15s} → {child:<18s}  *** NOT AVAILABLE ***")

        # Odom topic
        print()
        if self.last_odom:
            o = self.last_odom
            yaw = quat_to_yaw(o.pose.pose.orientation)
            print(f"  /odom  x={o.pose.pose.position.x:+7.3f}  y={o.pose.pose.position.y:+7.3f}  "
                  f"yaw={math.degrees(yaw):+7.1f}°  "
                  f"vx={o.twist.twist.linear.x:+5.3f}  wz={o.twist.twist.angular.z:+5.3f}  "
                  f"frame={o.header.frame_id}  child={o.child_frame_id}  "
                  f"(msgs: {self.odom_count})")
        else:
            print(f"  /odom  *** NO MESSAGES *** (msgs: {self.odom_count})")

        if self.last_wheel_odom:
            o = self.last_wheel_odom
            yaw = quat_to_yaw(o.pose.pose.orientation)
            print(f"  /wheel/odometry  x={o.pose.pose.position.x:+7.3f}  y={o.pose.pose.position.y:+7.3f}  "
                  f"yaw={math.degrees(yaw):+7.1f}°  "
                  f"vx={o.twist.twist.linear.x:+5.3f}  wz={o.twist.twist.angular.z:+5.3f}  "
                  f"(msgs: {self.wheel_odom_count})")
        else:
            print(f"  /wheel/odometry  *** NO MESSAGES *** (msgs: {self.wheel_odom_count})")

        # IMU
        print()
        if self.last_imu:
            i = self.last_imu
            yaw = quat_to_yaw(i.orientation)
            print(f"  /imu/data  yaw={math.degrees(yaw):+7.1f}°  "
                  f"gyro_z={i.angular_velocity.z:+6.4f} rad/s  "
                  f"frame={i.header.frame_id}  (msgs: {self.imu_count})")
        else:
            print(f"  /imu/data  *** NO MESSAGES *** (msgs: {self.imu_count})")

        # Scan
        print()
        if self.last_scan:
            s = self.last_scan
            valid = [r for r in s.ranges if s.range_min < r < s.range_max]
            print(f"  /scan  rays={len(s.ranges)}  valid={len(valid)}  "
                  f"min_range={min(valid) if valid else 0:.3f}m  "
                  f"frame={s.header.frame_id}  (msgs: {self.scan_count})")
        else:
            print(f"  /scan  *** NO MESSAGES *** (msgs: {self.scan_count})")

        # TF publisher check
        print()
        try:
            frames = self.tf_buffer.all_frames_as_yaml()
            # Parse which node publishes odom→base_footprint
            for block in frames.split('\n\n'):
                if 'base_footprint' in block and 'parent' in block and 'odom' in block:
                    print(f"  TF publisher info:\n    {block.strip()}")
        except:
            pass

        print(f"{'='*65}\n")

    def run(self):
        # Let data arrive
        self.get_logger().info("Collecting data for 3 seconds...")
        t0 = time.time()
        while time.time() - t0 < 3.0:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.print_snapshot("STATIONARY (t=0s)")

        if self.do_drive:
            # Drive forward 0.5s
            self.get_logger().info("Driving forward at 0.15 m/s for 1 second...")
            cmd = Twist()
            cmd.linear.x = 0.15
            t0 = time.time()
            while time.time() - t0 < 1.0:
                self.cmd_pub.publish(cmd)
                rclpy.spin_once(self, timeout_sec=0.05)
            self.cmd_pub.publish(Twist())  # stop

            time.sleep(0.5)
            # Collect
            for _ in range(15):
                rclpy.spin_once(self, timeout_sec=0.1)
            self.print_snapshot("AFTER DRIVING FORWARD ~0.15m (t=1s)")

            # Rotate left 0.5s
            self.get_logger().info("Rotating left at 0.5 rad/s for 1 second...")
            cmd = Twist()
            cmd.angular.z = 0.5
            t0 = time.time()
            while time.time() - t0 < 1.0:
                self.cmd_pub.publish(cmd)
                rclpy.spin_once(self, timeout_sec=0.05)
            self.cmd_pub.publish(Twist())  # stop

            time.sleep(0.5)
            for _ in range(15):
                rclpy.spin_once(self, timeout_sec=0.1)
            self.print_snapshot("AFTER ROTATING LEFT ~28° (t=2s)")
        else:
            self.get_logger().info("Run with -p drive_test:=true to also test driving.")
            self.get_logger().info("Monitoring for 10 more seconds (move robot manually with teleop)...")
            for i in range(10):
                t0 = time.time()
                while time.time() - t0 < 1.0:
                    rclpy.spin_once(self, timeout_sec=0.1)
                self.print_snapshot(f"t={i+1}s")


def main():
    rclpy.init()
    node = NavDiag()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
