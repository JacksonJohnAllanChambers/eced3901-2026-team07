#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import time

class RescueProcedure(Node):
    def __init__(self):
        super().__init__('rescue_procedure')

        self.ser = None

        # Serial connection to the CH340 Uno (udev symlink)
        port = '/dev/uart_arduino'
        try:
            self.ser = serial.Serial(port, 115200, timeout=0.1)
            self.get_logger().info(f'Connected to Arduino on {port}')
        except Exception as e:
            self.get_logger().error(f'Arduino Connection Failed: {e}')
            return

        # CH340 toggles DTR on connect which resets the Uno — wait for boot
        time.sleep(2)
        # Drain any startup messages from the Arduino
        while self.ser.in_waiting > 0:
            self.ser.readline()

        # Publisher for robot movement
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Step 1: Force a Reset to open the trap bar
        self.get_logger().info('Resetting Trap... Ensuring servos are open.')
        self.ser.write(b'R')
        time.sleep(1)  # Wait for mechanical movement
        # Drain the reset acknowledgement
        while self.ser.in_waiting > 0:
            self.ser.readline()

        # Logic state
        self.trapped = False
        self.get_logger().info('Starting search spin. Monitoring for Benchy...')

        # Timer to run the main loop at 10Hz
        self.timer = self.create_timer(0.1, self.run_logic)

    def run_logic(self):
        if self.ser is None:
            return
        if not self.trapped:
            # Command a slow spin (0.3 rad/s)
            move = Twist()
            move.angular.z = 0.3
            self.cmd_pub.publish(move)

            # Check if Arduino has triggered the trap
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='replace').strip()
                if 'TRAPPING' in line:
                    self.get_logger().info('TARGET ACQUIRED! Stopping robot.')
                    self.stop_robot()
                    self.trapped = True
                    self.get_logger().info('Benchy Secured. Send R to node to release.')

    def stop_robot(self):
        stop_move = Twist()
        self.cmd_pub.publish(stop_move)

def main(args=None):
    rclpy.init(args=args)
    node = RescueProcedure()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
