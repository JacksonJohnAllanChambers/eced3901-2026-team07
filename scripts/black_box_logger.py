#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import csv
import time
import math

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist

class BlackBoxRecorder(Node):
    def __init__(self):
        super().__init__('black_box_recorder')
        
        # Open a CSV file in the current directory
        self.filename = f"robot_diagnostic_{int(time.time())}.csv"
        self.file = open(self.filename, mode='w', newline='')
        self.writer = csv.writer(self.file)
        
        # Write Headers
        self.writer.writerow([
            'Time', 'Cmd_Vel_X', 'Cmd_Vel_Z',
            'IMU_Yaw_Deg', 
            'Odom_X', 'Odom_Y', 'Odom_Yaw_Deg',
            'AMCL_X', 'AMCL_Y', 'AMCL_Yaw_Deg'
        ])

        # Data storage variables
        self.cmd_v = 0.0
        self.cmd_w = 0.0
        self.imu_yaw = 0.0
        self.odom_x, self.odom_y, self.odom_yaw = 0.0, 0.0, 0.0
        self.amcl_x, self.amcl_y, self.amcl_yaw = 0.0, 0.0, 0.0

        # Subscriptions
        self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)
        self.create_subscription(Imu, '/bno055/imu', self.imu_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.amcl_cb, 10)

        # Timer to write to CSV at 20Hz (every 0.05 seconds)
        self.timer = self.create_timer(0.05, self.record_data)
        self.get_logger().info(f"🔴 Recording Black Box Data to: {self.filename}")

    # --- Helper to convert Quaternions to Degrees ---
    def get_yaw(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        return math.degrees(yaw_rad)

    # --- Callbacks update the latest data ---
    def cmd_cb(self, msg):
        self.cmd_v = msg.linear.x
        self.cmd_w = msg.angular.z

    def imu_cb(self, msg):
        self.imu_yaw = self.get_yaw(msg.orientation)

    def odom_cb(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_yaw = self.get_yaw(msg.pose.pose.orientation)

    def amcl_cb(self, msg):
        self.amcl_x = msg.pose.pose.position.x
        self.amcl_y = msg.pose.pose.position.y
        self.amcl_yaw = self.get_yaw(msg.pose.pose.orientation)

    # --- Write current snapshot to CSV ---
    def record_data(self):
        # Record time relative to the start of the script for easier reading
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()
        
        t = time.time() - self.start_time
        
        self.writer.writerow([
            f"{t:.3f}", 
            f"{self.cmd_v:.3f}", f"{self.cmd_w:.3f}",
            f"{self.imu_yaw:.2f}",
            f"{self.odom_x:.3f}", f"{self.odom_y:.3f}", f"{self.odom_yaw:.2f}",
            f"{self.amcl_x:.3f}", f"{self.amcl_y:.3f}", f"{self.amcl_yaw:.2f}"
        ])

    def destroy_node(self):
        self.file.close()
        super().destroy_node()

def main():
    rclpy.init()
    node = BlackBoxRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Stopping recording and saving file.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
