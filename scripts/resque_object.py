#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32, Empty
import math
import time

class RescueObjectOdom(Node):
    def __init__(self):
        super().__init__('rescue_object_odom')
        
        # Publishers
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.servo_pub = self.create_publisher(Empty, '/servo_trigger', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.tof_sub = self.create_subscription(Int32, '/tof_distance', self.tof_callback, 10)
        
        # Logic state
        self.current_yaw = 0.0
        self.min_dist = 2000
        self.best_yaw = 0.0
        self.is_scanning = False

    def odom_callback(self, msg):
        # Convert Quaternion to Yaw (Z-axis rotation)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def tof_callback(self, msg):
        if self.is_scanning:
            if 0 < msg.data < self.min_dist:
                self.min_dist = msg.data
                self.best_yaw = self.current_yaw
                self.get_logger().info(f'Target at {math.degrees(self.best_yaw):.2f}°')

    def rotate_to(self, target_yaw):
        move = Twist()
        kp = 0.8  # Proportional gain for turning
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            error = target_yaw - self.current_yaw
            
            # Keep error within [-pi, pi]
            error = math.atan2(math.sin(error), math.cos(error))
            
            if abs(error) < 0.05:  # Tolerance (~3 degrees)
                break
                
            move.angular.z = kp * error
            self.vel_pub.publish(move)
            
        move.angular.z = 0.0
        self.vel_pub.publish(move)

    def run_mission(self):
        time.sleep(2.0)
        self.get_logger().info('Scanning 360 degrees...')
        
        start_yaw = self.current_yaw
        self.is_scanning = True
        
        # Rotate until we have completed a full circle
        move = Twist()
        move.angular.z = 0.5
        while rclpy.ok():
            self.vel_pub.publish(move)
            rclpy.spin_once(self, timeout_sec=0.05)
            # Check if we've returned close to our start point after moving
            if abs(self.current_yaw - start_yaw) < 0.1 and self.min_dist < 2000:
                # Add a small delay to ensure we actually started moving first
                break
                
        self.is_scanning = False
        self.get_logger().info('Alignment phase...')
        self.rotate_to(self.best_yaw)
        
        self.get_logger().info('Trapping target!')
        self.servo_pub.publish(Empty())

def main():
    rclpy.init()
    node = RescueObjectOdom()
    node.run_mission()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
