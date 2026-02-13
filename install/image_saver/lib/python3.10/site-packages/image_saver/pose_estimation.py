import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import Header, ColorRGBA
from transforms3d import quaternions as quat_math
from cv_bridge import CvBridge
from .rgb_detection import rgb_detector
import numpy as np
import cv2
import yaml



class PoseEstimateNode(Node):
    def __init__(self,name):
        super().__init__(name)
        self.get_logger().info(f'{name} loaded successfully!')

        self.bridge = CvBridge()
        self.image_subscriber = self.create_subscription(Image,f'/camera1/image_raw',\
                                                         self.estimation_callback,10)
        self.marker_publisher = self.create_publisher(Marker,'/visualization_marker',10)

        self.square_size = 0.075
        half_size = self.square_size/2
        self.objp_3d = np.array([
            [-half_size, -half_size, 0],
            [half_size, -half_size, 0],
            [half_size, half_size, 0],
            [-half_size, half_size, 0],
        ], dtype=np.float32)

        self.axis_length = 0.05
        self.axis_3d = np.array([
            [self.axis_length, 0, 0],
            [0, self.axis_length, 0],
            [0, 0, self.axis_length],
        ], dtype=np.float32)

        self.calibration_path = 'src/image_saver/image_saver/config/ost.yaml'
        self.mtx,self.dist = self.load_calibration_params()

        self.marker_id = 0
        

    def load_calibration_params(self):
        mtx = np.array([
            [618.866180, 0.0, 335.210126],
            [0.0, 618.965733, 232.816692],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        dist = np.array([-0.465789, 0.215461, -0.004792, -0.000580, 0.000000], dtype=np.float32)
        
        return mtx, dist
    

    def rotation_vector_to_quaternion(self,rvec):
        rotation_matrix,_ = cv2.Rodrigues(rvec)
        q = quat_math.mat2quat(rotation_matrix)
        return Quaternion(x=float(q[1]),y=float(q[2]),z=float(q[3]),w=float(q[0]))
    

    def create_cube_marker(self,rvec,tvec):
        marker = Marker()

        marker.header = Header()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'camera1_link'

        marker.ns = "detected_cube"
        marker.id = self.marker_id

        marker.type = Marker.CUBE

        marker.action = Marker.ADD

        pose_msg = Pose()

        pose_msg.position.x = float(tvec[0])
        pose_msg.position.y = float(tvec[1])
        pose_msg.position.z = float(tvec[2])
        
        pose_msg.orientation = self.rotation_vector_to_quaternion(rvec)
        marker.pose = pose_msg

        display_scale = 25.0
        marker.scale = Vector3()
        marker.scale.x = self.square_size * display_scale
        marker.scale.y = self.square_size * display_scale
        marker.scale.z = self.square_size * display_scale

        marker.color = ColorRGBA()
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()

        return marker

    def estimate_pose(self,corner):
        if corner is None:
            return None,None,None
        
        success, rvec, tvec = cv2.solvePnP(
            self.objp_3d,
            corner,
            self.mtx,
            self.dist,
            flags = cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            self.get_logger().info(f"Estimation succeeded!")
            t = tvec.flatten()
            self.get_logger().info(f"   pos ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")

            marker = self.create_cube_marker(rvec,tvec)
            self.marker_publisher.publish(marker)
            self.get_logger().info(f'Marker published')

            axis_2d,_ = cv2.projectPoints(self.axis_3d,rvec,tvec,self.mtx,self.dist)

            return rvec,tvec,axis_2d
        
        else:
            self.get_logger().error("Estimation failed...")
            return None,None,None
        

    def draw_axes_on_image(self, img, corner, axis_2d,rvec,tvec):
        try:
            center_3d = np.array([[0, 0, 0]], dtype=np.float32)
            center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, self.mtx, self.dist)
            origin = tuple(map(int, center_2d[0][0]))
            
            x_axis = tuple(map(int, axis_2d[0].ravel()))
            y_axis = tuple(map(int, axis_2d[1].ravel()))
            z_axis = tuple(map(int, axis_2d[2].ravel()))
            
            img = cv2.line(img, origin, x_axis, (255,0,0), 5)
            img = cv2.line(img, origin, y_axis, (0,255,0), 5)
            img = cv2.line(img, origin, z_axis, (0,0,255), 5)

            cv2.circle(img,origin,5,(0,255,255),-1)

        except Exception as e:
            self.get_logger().error(f'Draw axes failed: {e}')
        
        return img


    def estimation_callback(self,msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg,'bgr8')
        except Exception as e:
            self.get_logger().error(f'ROS img message transformation failed:{e}')
            return
        
        try:
            detector = rgb_detector()
            img,corner_list = detector.detect(cv_image,'green')
            if corner_list and len(corner_list)>0:
                corner = max(corner_list,key=lambda c:(c[2][0]-c[0][0])*(c[2][1]-c[0][1]))

                rvec,tvec,axis_2d = self.estimate_pose(corner)
                if axis_2d is not None:
                    img = self.draw_axes_on_image(img,corner,axis_2d,rvec,tvec)

                cv2.imshow('Pose Estimation',img)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Image process failed:{e}')


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimateNode(f'estimator')

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f'User interruption')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()