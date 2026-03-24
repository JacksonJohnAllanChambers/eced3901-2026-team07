import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import threading
import os

class FrameCapture(Node):
    def __init__(self):
        super().__init__('frame_capture')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.create_subscription(Image, '/camera/image_raw', self.img_cb, 10)
        
        self.save_dir = "captured_frames"
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"Frame capture ready. Saving to: {self.save_dir}/")
        
        self.capture_thread = threading.Thread(target=self.input_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def img_cb(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def input_loop(self):
        count = 1
        print("\n" + "="*40)
        print("📷 FRAME CAPTURE SCRIPT")
        print("Press ENTER in this terminal to safely capture")
        print("the robot's current camera frame to disk.")
        print("Type 'q' + ENTER to quit.")
        print("="*40 + "\n")
        
        while True:
            try:
                cmd = input()
                if cmd.strip().lower() == 'q':
                    print("Exiting...")
                    os._exit(0)
                
                if self.latest_frame is not None:
                    # Save a copy to prevent thread race conditions
                    frame_copy = self.latest_frame.copy()
                    fname = f"{self.save_dir}/angled_view_{count:03d}.jpg"
                    cv2.imwrite(fname, frame_copy)
                    print(f"✅ Saved {fname}!")
                    count += 1
                else:
                    print("❌ No frame received yet. Check /camera/image_raw!")
            except (EOFError, KeyboardInterrupt):
                os._exit(0)

def main():
    rclpy.init()
    node = FrameCapture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
