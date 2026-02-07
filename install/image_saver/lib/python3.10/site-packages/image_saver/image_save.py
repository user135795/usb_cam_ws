import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2

class constant:
    BGR_MAX = 255

class Subscriber00(Node):

    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info(f'{name} loaded successfully!')

        self.bridge = CvBridge()
        self.save_count = 0
        self.image_subscriber = self.create_subscription(Image,f'/camera1/image_raw',self.image_callback,10)


    def image_pixel_get(self,img,y,x):
        pixel_value = img[y,x]
        return pixel_value
    

    def image_pixel_alter(self,img,y,x,b,g,r):
        img[y,x] = [b,r,g]


    def image_split(self,img):
        blue,green,red = cv2.split(img)
        return [blue,green,red]


    def image_merge(self,blue,green,red):
        merged_image = cv2.merge([blue,green,red])
        return merged_image
    

    def display_color(self,img,b,g,r):
        zeros = np.zeros_like(b)

        blue_colored = self.image_merge(b,zeros,zeros)
        green_colored = self.image_merge(zeros,g,zeros)
        red_colored = self.image_merge(zeros,zeros,r)

        cv2.imshow('Blue Channel',blue_colored)
        cv2.waitKey(2000)
        cv2.destroyWindow('Blue Channel')
        cv2.imshow('Green Channel',green_colored)
        cv2.waitKey(2000)
        cv2.destroyWindow('Green Channel')
        cv2.imshow('Red Channel',red_colored)
        cv2.waitKey(2000)
        cv2.destroyWindow('Red Channel')


    def display_color_plus(self,img,b,g,r):
        gray_value = (g.astype(np.float32) + r.astype(np.float32)) / 2
        gray_value = gray_value.astype(np.uint8)
        blue_colored = self.image_merge(b,gray_value,gray_value)
        cv2.imshow('Blue Channel',blue_colored)
        cv2.waitKey(2000)
        cv2.destroyWindow('Blue Channel')

        gray_value = (r.astype(np.float32) + b.astype(np.float32)) / 2
        gray_value = gray_value.astype(np.uint8)
        green_colored = self.image_merge(gray_value,g,gray_value)
        cv2.imshow('Green Channel',green_colored)
        cv2.waitKey(2000)
        cv2.destroyWindow('Green Channel')

        gray_value = (g.astype(np.float32) + b.astype(np.float32)) / 2
        gray_value = gray_value.astype(np.uint8)
        red_colored = self.image_merge(gray_value,gray_value,r)
        cv2.imshow('Red Channel',red_colored)
        cv2.waitKey(2000)
        cv2.destroyWindow('Red Channel')

    def image_callback(self,msg):
        self.get_logger().info(f'收到图像消息: 宽度={msg.width}, 高度={msg.height}')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg,'bgr8')

        except Exception as e:
            self.get_logger().error(f'转换ROS图像消息失败: {e}')

        else:
            self.get_logger().info(f'transformation success!')

            cv2.imshow(f'USB camera',cv_image)

            splited_image = self.image_split(cv_image)
            self.display_color_plus(cv_image,splited_image[0],splited_image[1],splited_image[2])

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                self.save_count += 1
                filename = f'image{self.save_count:04d}.jpg'

                if cv2.imwrite(f'saved_images/{filename}',cv_image):
                    self.get_logger().info(f'{filename} saved successfully!')
                    cv2.destroyAllWindows()
                else:
                    self.get_logger().error(f'Failed to save: {filename}')


def main(args=None):
    rclpy.init(args=args)
    node = Subscriber00(f'subscriber')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f'用户中断程序')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()