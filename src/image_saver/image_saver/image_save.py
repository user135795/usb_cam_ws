import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
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

    def image_callback(self,msg):
        self.get_logger().info(f'收到图像消息: 宽度={msg.width}, 高度={msg.height}')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg,'bgr8')

        except Exception as e:
            self.get_logger().error(f'转换ROS图像消息失败: {e}')

        else:
            self.get_logger().info(f'transformation success!')

            for i in range(0,msg.width):
                for j in range(0,msg.height):
                    pixel_value = self.image_pixel_get(cv_image,j,i)
                    self.image_pixel_alter(cv_image,j,i,(pixel_value[0]+100)%constant.BGR_MAX\
                                           ,(pixel_value[1]+100)%constant.BGR_MAX,\
                                            (pixel_value[2]+100)%constant.BGR_MAX)

            cv2.imshow(f'USB camera',cv_image)
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