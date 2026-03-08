#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import subprocess
import time
import os

class MapSaverNode(Node):
    def __init__(self):
        super().__init__('map_saver_node')
        
        # Configuration
        # Default wait time is 50 seconds (Adjust this based on your robot's speed)
        # 1m x 4 sides / 0.1 m/s = ~40 seconds + turning time.
        self.declare_parameter('save_delay', 50.0) 
        self.declare_parameter('map_name', 'lab5_dt2_map')
        
        self.save_delay = self.get_parameter('save_delay').value
        self.map_name = self.get_parameter('map_name').value
        
        # Timer to trigger the save
        self.get_logger().info(f'Map Saver initialized. Waiting {self.save_delay} seconds to save map...')
        self.timer = self.create_timer(self.save_delay, self.save_map_callback)

    def save_map_callback(self):
        # We only want to run this once, so cancel the timer immediately
        self.timer.cancel()
        
        self.get_logger().info('Time is up! Attempting to save the map...')
        
        # Define the save path
        # This saves to your package's map folder so it is easy to find later
        home_dir = os.path.expanduser('~')
        save_dir = os.path.join(home_dir, 'ros2_ws', 'src', 'eced3901', 'maps')
        
        # Ensure directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, self.map_name)
        
        # Command to run the Nav2 map saver
        # "ros2 run nav2_map_server map_saver_cli -f <path_to_map>"
        command = ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', save_path]
        
        try:
            self.get_logger().info(f"Executing: {' '.join(command)}")
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.get_logger().info(f'SUCCESS: Map saved to {save_path}')
                self.get_logger().info(result.stdout)
            else:
                self.get_logger().error('FAILURE: Could not save map.')
                self.get_logger().error(result.stderr)
                
        except Exception as e:
            self.get_logger().error(f'An error occurred: {str(e)}')
            
        # Shutdown this node after the attempt so the launch file can exit cleanly if needed
        # rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = MapSaverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()