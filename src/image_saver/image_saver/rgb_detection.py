import rclpy
from rclpy.node import Node
import numpy as np
import cv2

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

class rgb_detector(Node):
    def __init__(self):
        super().__init__('rgb_detector')
    
    def detect(self,img,target_color):
        blurred_img = cv2.GaussianBlur(img,(5,5),0)
        hsv_img = cv2.cvtColor(blurred_img,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img,constant.color_dist[target_color]['lower'],\
                            constant.color_dist[target_color]['upper'])
        
        kernal = np.ones((5,5),"uint8")
        mask = cv2.dilate(mask,kernal)

        contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if(hierarchy is not None):
            hierarchy = hierarchy[0]

        corner_list = []

        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 5000) and (hierarchy is None or hierarchy[pic][3]==-1):
                x, y, w, h = cv2.boundingRect(contour)

                corner = np.array([
                    [x,y],
                    [x+w, y],
                    [x+w, y+h],
                    [x,y+h],
                ], dtype = np.float32)

                corner_list.append(corner)

                self.get_logger().info(f'{target_color} contour{pic}:x:{x},y:{y},w:{w},h:{h}')
                img = cv2.rectangle(img,(x,y),(x+w,y+h),constant.color_dist[target_color]['value'],2)
                cv2.putText(img,target_color,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1.0,constant.color_dist[target_color]['value'])
        
        return img,corner_list
