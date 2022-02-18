#!/usr/bin/env python3

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Python imports
import numpy as np
import cv2
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf
# ROS imports
import rospy
import scipy.io as sio
import std_msgs.msg
# Deep Learning imports
import torch
import yaml
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point32, Polygon
from numpy import random
from rospkg import RosPack
from sensor_msgs.msg import Image
from skimage.transform import resize
from std_msgs.msg import UInt8
from torch.autograd import Variable
from yolov5_pytorch_ros.msg import BoundingBox, BoundingBoxes

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import (apply_classifier, check_img_size,
                           check_requirements, increment_path,
                           non_max_suppression, scale_coords, set_logging,
                           strip_optimizer)
from utils.plots import plot_one_box
# util + model imports
from utils.torch_utils import load_classifier, select_device, time_synchronized

package = RosPack()
package_path = package.get_path('yolov5_pytorch_ros')
topic_tf_child = "/object"
topic_tf_perent = "/base_link"
t = TransformStamped()
tf2_br = tf2_ros.TransformBroadcaster()

class Detector:
    def __init__(self):
        # Load weights parameter
        self.weights_path = rospy.get_param('~weights_path')
        rospy.loginfo("Found weights, loading %s", self.weights_path)

        # Raise error if it cannot find the model
        if not os.path.isfile(self.weights_path):
            raise IOError(('{:s} not found.').format(self.weights_path))

        # Load image parameter and confidence threshold
        self.image_topic = rospy.get_param(
            '~image_topic', '/camera/rgb/image_raw')

        log_str = "Subscribing to image topic: %s" % self.image_topic
        rospy.loginfo(log_str)

        self.conf_thres = rospy.get_param('~confidence', 0.25)

        # Load other parameters
        self.device_name = ''
        self.device = select_device(self.device_name)
        self.gpu_id = rospy.get_param('~gpu_id', 0)
        self.network_img_size = rospy.get_param('~img_size', 416)
        self.publish_image = rospy.get_param('~publish_image')
        self.iou_thres = 0.45
        self.augment = True

        self.classes = None
        self.agnostic_nms = False

        self.w = 0
        self.h = 0

        # Second-stage classifier
        self.classify = False

        # Initialize
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(
            self.weights_path, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())      # model stride
        self.network_img_size = check_img_size(
            self.network_img_size, s=self.stride)  # check img_size

        if self.half:
            self.model.half()  # to FP16

        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load(
                'weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Get names and colors
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255)
                        for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.network_img_size, self.network_img_size).to(
                self.device).type_as(next(self.model.parameters())))  # run once
        self.K=[203.42144086256206, 0.0, 206.12517266886093, 0.0, 203.55319738398958, 146.4392791209304, 0.0, 0.0, 1.0]
        t = 0
        self.camera_parameters=np.array([[0,0,0],[0,0,0],[0,0,0]])
        for i in range(3):
            for k in range(3):
                self.camera_parameters[i][k] = self.K[i+k]
                t+=1
        self.D=[-0.18296735250090237, 0.49168852367941696, -0.6266727991904993, 0.2636407064533411,0]
        self.camera_distortion_param = np.array([-0.426801, 0.249082, -0.002705, -0.001600, 0.000000])
        # Load CvBridge
        self.bridge = CvBridge()

        # Load publisher topic
        self.detected_objects_topic = rospy.get_param(
            '~detected_objects_topic')
        self.published_image_topic = rospy.get_param('~detections_image_topic')

        # Define subscribers
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)
        self.image_sub = rospy.Subscriber(
            self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)
        # Define publishers
        self.pub_ = rospy.Publisher(
            self.detected_objects_topic, BoundingBoxes, queue_size=10)
        self.pub_viz_ = rospy.Publisher(
            self.published_image_topic, Image, queue_size=10)
        rospy.loginfo("Launched node for object detection")

        # Spin
        rospy.spin()
    def pubTf(self,position, orientation):
        """
        publish find object to tf2
        :param position:
        :param orientation:
        :return:
        """
        global t
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = topic_tf_perent
        t.child_frame_id = topic_tf_child
        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]
        quaternion = tf.transformations.quaternion_from_euler(orientation[0], orientation[1], orientation[2])

        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        tf2_br.sendTransform(t)

    def iterative_solve_pnp(self, object_points, image_points, camera_parameters, camera_distortion_param):
        image_points = image_points.reshape(-1, 2)
        retval, rotation, translation = cv2.solvePnP(object_points, image_points, camera_parameters,
                                                     camera_distortion_param)
        return rotation, translation
    def image_cb(self, data):
        # Convert the image to OpenCV
        try:
            self.cv_img = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)
        # Initialize detection results

        detection_results = BoundingBoxes()
        detection_results.header = data.header
        detection_results.image_header = data.header
        input_img = self.preprocess(self.cv_img)
        #input_img = Variable(input_img.type(torch.FloatTensor))

        # Get detections from network
        with torch.no_grad():
            input_img = torch.from_numpy(input_img).to(self.device,dtype=torch.half)
            detections = self.model(input_img)[0]
            detections = non_max_suppression(detections, self.conf_thres, self.iou_thres,
                                             classes=self.classes, agnostic=self.agnostic_nms)
            
        # Parse detections
        if detections[0] is not None:
            for detection in detections[0]:
                
                # Get xmin, ymin, xmax, ymax, confidence and class
                xmin, ymin, xmax, ymax, conf, det_class = detection
        
                if self.names[int(det_class)] not in ['bin_rect','bin_side','bin_cir']:
                    continue
                pad_x = max(self.h - self.w, 0) * \
                    (self.network_img_size/max(self.h, self.w))
                pad_y = max(self.w - self.h, 0) * \
                    (self.network_img_size/max(self.h, self.w))
                unpad_h = self.network_img_size-pad_y
                unpad_w = self.network_img_size-pad_x
                xmin_unpad = ((xmin-pad_x//2)/unpad_w)*self.w
                xmax_unpad = ((xmax-xmin)/unpad_w)*self.w + xmin_unpad
                ymin_unpad = ((ymin-pad_y//2)/unpad_h)*self.h
                ymax_unpad = ((ymax-ymin)/unpad_h)*self.h + ymin_unpad
                xmin_unpad = xmin_unpad.cpu().detach().numpy()
                xmax_unpad = xmax_unpad.cpu().detach().numpy()
                ymin_unpad =  ymin_unpad.cpu().detach().numpy()
                ymax_unpad = ymax_unpad.cpu().detach().numpy()
                # Populate darknet message
                detection_msg = BoundingBox()
                detection_msg.xmin = int(xmin_unpad)
                detection_msg.xmax = int(xmax_unpad)
                detection_msg.ymin = int(ymin_unpad)
                detection_msg.ymax = int(ymax_unpad)
                detection_msg.probability = float(conf)
                detection_msg.Class = self.names[int(det_class)]
                object=np.array([[xmin_unpad,ymin_unpad,0],
                                [xmax_unpad,ymin_unpad,0],
                                [xmin_unpad,ymax_unpad,0],
                                [xmax_unpad,ymax_unpad,0]],dtype=np.float64)
                image=np.array([[xmin_unpad,ymin_unpad],
                                [xmax_unpad,ymin_unpad],
                                [xmin_unpad,ymax_unpad],
                                [xmax_unpad,ymax_unpad]],dtype=np.float32)
                rotation_rad, translation = self.iterative_solve_pnp(object,image,
                                                                     self.camera_parameters,
                                                                     self.camera_distortion_param)
                
                w=((xmax.cpu().detach().numpy()-xmin.cpu().detach().numpy())/640)
                print(w)
                translation= w*translation
                self.pubTf(translation,rotation_rad)
                # Append in overall detection message
                detection_results.bounding_boxes.append(detection_msg)

        # Publish detection results
        self.pub_.publish(detection_results)

        # Visualize detection results
        if (self.publish_image):
            self.visualize_and_publish(detection_results, self.cv_img)
        return True

    def preprocess(self, img):
        # Extract image and shape
        img = np.copy(img)
        img = img.astype(float)
        height, width, channels = img.shape
        if (height != self.h) or (width != self.w):
            self.h = height
            self.w = width
            # Determine image to be used
            self.padded_image = np.zeros(
                (max(self.h, self.w), max(self.h, self.w), channels)).astype(float)

        # Add padding
        if (self.w > self.h):
            self.padded_image[(self.w-self.h)//2: self.h +
                              (self.w-self.h)//2, :, :] = img
        else:
            self.padded_image[:, (self.h-self.w)//2: self.w +
                              (self.h-self.w)//2, :] = img
        # Resize and normalize
        input_img = resize(self.padded_image, (self.network_img_size, self.network_img_size, 3))/255.

        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))

        # As pytorch tensor
        #input_img = torch.from_numpy(input_img).float()
        input_img = input_img[None]

        return input_img

    def visualize_and_publish(self, output, imgIn):
        # Copy image and visualize
        imgOut = imgIn.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        thickness = 2
        for index in range(len(output.bounding_boxes)):
            label = output.bounding_boxes[index].Class
            x_p1 = output.bounding_boxes[index].xmin
            y_p1 = output.bounding_boxes[index].ymin
            x_p3 = output.bounding_boxes[index].xmax
            y_p3 = output.bounding_boxes[index].ymax
            confidence = output.bounding_boxes[index].probability

            # Set class color
            color = self.colors[self.names.index(label)]

            # Create rectangle
            cv2.rectangle(imgOut, (int(x_p1), int(y_p1)), (int(x_p3), int(
                y_p3)), (color[0], color[1], color[2]), thickness)
            text = ('{:s}: {:.3f}').format(label, confidence)
            cv2.putText(imgOut, text, (int(x_p1), int(y_p1+20)), font,
                        fontScale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Publish visualization image
        image_msg = self.bridge.cv2_to_imgmsg(imgOut, "rgb8")
        image_msg.header.frame_id = 'camera'
        image_msg.header.stamp = rospy.Time.now()
        self.pub_viz_.publish(image_msg)


if __name__ == '__main__':
    rospy.init_node('detector')

    # Define detector object
    dm = Detector()
