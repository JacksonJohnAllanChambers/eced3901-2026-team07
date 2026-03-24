#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import csv
import math
import time

class CoastalTrajectoryLogger(Node):
    def __init__(self):
        super().__init__('coastal_trajectory_logger')
        
        # Subscribe to the real robot's EKF odometry
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
            
        self.log_interval = 0.1  # Log at 10 Hz
        self.last_log_time = time.time()
        
        # Explicitly name the CSV for Coastal tracking
        self.filename = f"real_coastal_trajectory_{int(time.time())}.csv"
        self.get_logger().info(f"Coastal Logger Started! Saving to: {self.filename}")
        
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'X', 'Y', 'Yaw_Deg'])

    def euler_from_quaternion(self, x, y, z, w):
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return yaw_z

    def odom_callback(self, msg):
        current_time = time.time()
        
        if current_time - self.last_log_time >= self.log_interval:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            
            q = msg.pose.pose.orientation
            yaw_deg = math.degrees(self.euler_from_quaternion(q.x, q.y, q.z, q.w))
            
            with open(self.filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([current_time, x, y, yaw_deg])
                
            self.last_log_time = current_time

def main(args=None):
    rclpy.init(args=args)
    logger = CoastalTrajectoryLogger()
    
    try:
        rclpy.spin(logger)
    except KeyboardInterrupt:
        logger.get_logger().info(f"Data logging stopped. Saved to {logger.filename}")
    finally:
        logger.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
