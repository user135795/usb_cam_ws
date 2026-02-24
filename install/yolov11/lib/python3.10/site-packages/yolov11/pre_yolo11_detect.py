import sys
import os
conda_site_packages = '/home/cherryad3/miniconda3/envs/YOLO11/lib/python3.10/site-packages'
if conda_site_packages not in sys.path:
    sys.path.insert(0,conda_site_packages)
ultralytics_path = '/home/cherryad3/YOLO11/ultralytics-main'
if os.path.exists(ultralytics_path) and ultralytics_path not in sys.path:
    sys.path.insert(0, ultralytics_path)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO


class yolo11_detector1(Node):

    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info(f'{name} loaded successfully!')

        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(Image,f'/camera1/image_raw',\
                                                         self.detect_callback,10)
        self.model = YOLO("src/yolov11/yolo11n.pt")

    def detect_callback(self,msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg,'bgr8')
        except Exception as e:
            self.get_logger().error(f'ROS img message transformation failed: {e}')
        
        try:
            results = self.model(cv_image)
        except Exception as e:
            self.get_logger().error(f'detection failed: {e}')
        else:
            for result in results:
                plotted = result.plot()
                cv2.imshow("detect_result",plotted)
            cv2.waitKey(1)
        


def main(args=None):
    rclpy.init(args=args)
    node = yolo11_detector1('detector1')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('user interruption')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()



