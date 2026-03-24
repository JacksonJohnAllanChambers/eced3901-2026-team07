import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial

class UARTNode(Node):
    def __init__(self):
        super().__init__('uart_node')

        # Publishers / Subscribers
        self.rx_pub = self.create_publisher(String, 'uart_rx', 10)
        self.create_subscription(String, 'uart_tx', self._tx_cb, 10)

        # Serial port
        self.ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.1)

        # Poll serial at 50Hz
        self.create_timer(0.02, self._read_serial)

    def _read_serial(self):
        if self.ser.in_waiting > 0:
            data = self.ser.readline().decode('utf-8').strip()
            msg = String()
            msg.data = data
            self.rx_pub.publish(msg)
            self.get_logger().info(f'RX: {data}')

    def _tx_cb(self, msg):
        self.ser.write(msg.data.encode('utf-8'))
        self.get_logger().info(f'TX: {msg.data}')


def main():
    rclpy.init()
    node = UARTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
