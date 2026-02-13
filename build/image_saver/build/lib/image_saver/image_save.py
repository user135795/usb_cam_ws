import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from .rgb_detection import rgb_detector
class constant:
    BGR_MAX = 255
    color_dist = {
        'red1': {
            'lower': np.array([0, 180, 150], np.uint8),
            'upper': np.array([8, 255, 255], np.uint8),
            'value': (0, 0, 255)
        },
        'red2': {
            'lower': np.array([170, 100, 50], np.uint8),
            'upper': np.array([179, 255, 200], np.uint8),
            'value': (0, 0, 255)
        },
        'blue': {
            'lower': np.array([100, 100, 80], np.uint8),
            'upper': np.array([125, 255, 255], np.uint8),
            'value': (255, 0, 0)
        },
        'green': {
            'lower': np.array([45, 100, 80], np.uint8),
            'upper': np.array([75, 255, 255], np.uint8),
            'value': (0, 255, 0)
        }
    }


class Subscriber00(Node):

    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info(f'{name} loaded successfully!')

        self.bridge = CvBridge()
        self.save_count = 0
        self.image_subscriber = self.create_subscription(Image,f'/camera1/image_raw',\
                                                         self.image_callback,10)


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
    

    #display one color
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


    #emphasize one color,turning the others into gray
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


    def image_gray_convert(self,img):
        gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return gray_image
    

    def image_hsv_convert(self,img):
        hsv_image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        return hsv_image
    

    def image_binary_threshold(self,img):
        gray_image = self.image_gray_convert(img)
        retval,thresh_binary = cv2.threshold(gray_image,120,255,cv2.THRESH_BINARY)
        return retval,thresh_binary


    # def rgb_detection(self,img,target_color):
    #     blurred_img = cv2.GaussianBlur(img,(5,5),0)
    #     hsv_img = cv2.cvtColor(blurred_img,cv2.COLOR_BGR2HSV)
    #     mask = cv2.inRange(hsv_img,constant.color_dist[target_color]['lower'],\
    #                       constant.color_dist[target_color]['upper'])
        
    #     kernal = np.ones((5,5),"uint8")
    #     mask = cv2.dilate(mask,kernal)

    #     contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #     if(hierarchy is not None):
    #         hierarchy = hierarchy[0]

    #     for pic, contour in enumerate(contours):
    #         area = cv2.contourArea(contour)
    #         if(area > 300) and (hierarchy is None or hierarchy[pic][3]==-1):
    #             x, y, w, h = cv2.boundingRect(contour)
    #             self.get_logger().info(f'{target_color} contour{pic}:x:{x},y:{y},w:{w},h:{h}')
    #             img = cv2.rectangle(img,(x,y),(x+w,y+h),constant.color_dist[target_color]['value'],2)
    #             cv2.putText(img,target_color,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,constant.color_dist[target_color]['value'])
        
    #     return img
    
    def image_callback(self,msg):
        #self.get_logger().info(f'image received: width={msg.width}, height={msg.height}')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg,'bgr8')

        except Exception as e:
            self.get_logger().error(f'ROS img message transformation failed: {e}')

        else:
            #self.get_logger().info(f'transformation success!')
            # detector = rgb_detector()
            # res = detector.detect(cv_image,'green')
            # res = detector.detect(res,'blue')
            # res = detector.detect(res,'red1')
            # res = detector.detect(res,'red2')
            # cv2.imshow('rgb detection result',res)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                self.save_count += 1
                filename = f'image{self.save_count:04d}.jpg'

                if cv2.imwrite(f'saved_images/{filename}',res):
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
        node.get_logger().info(f'user interruption')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()