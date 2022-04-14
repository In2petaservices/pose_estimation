#!/usr/bin/env python3

import os
from tarfile import XHDTYPE
from this import d
from cv2 import KeyPoint
from matplotlib import image
import csv
import statistics
import image_processing
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import triangulation as tri
import calibration
# Python imports
import numpy as np
import cv2
import math
import message_filters
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf
from utils_ds.parser import get_config
from utils_ds.draw import draw_boxes
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
from sensor_msgs.msg import Image ,LaserScan
from skimage.transform import resize
from std_msgs.msg import UInt8
from torch.autograd import Variable
from yolov5_pytorch_ros.msg import BoundingBox, BoundingBoxes
from math import nan
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import (apply_classifier, check_img_size,
                           check_requirements, increment_path,
                           non_max_suppression, scale_coords, set_logging,
                           strip_optimizer, xyxy2xywh)
from utils.plots import plot_one_box
# util + model imports
from utils.torch_utils import load_classifier, select_device, time_synchronized
from deep_sort import build_tracker
from centroidtracker import CentroidTracker
package = RosPack()
package_path = package.get_path('yolov5_pytorch_ros')
topic_tf_child = "/object"
topic_tf_perent = "/base_link"
P = np.matrix([[0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0]])
x_hat=None
u=0
prev=[0,0]
intila=0
n=0
t = TransformStamped()
tf2_br = tf2_ros.TransformBroadcaster()
error_sum=0
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
        self.C=np.matrix([[1,0,1,0],
                            [0,1,0,1],
                            [0,0,0,0],
                            [0,0,0,0]])
        self.B=np.array([0,0,0,0])
        dt=0.2
        self.A=np.matrix([[1.0 ,0, dt,0],
                        [0,1.0,0,dt],
                        [0,0,1,0],
                        [0,0,0,1]])
        self.Q=np.matrix([[1.0,0,0,0],
                        [0,1.0,0,0],
                        [0,0,0.1,0],
                        [0,0,0,0.1]])
        self.R=np.matrix([[0.1,0,1,0],
                            [0,0.1,0,1],
                            [0,0,0.1,0],
                            [0,0,0,0.1]])
        self.I=np.matrix([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
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
        cfg = get_config()
        self.deepsort = build_tracker()
        cfg = get_config()
        self.deepsort2 = build_tracker()
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
            print(self.model)
            self.model(torch.zeros(1, 3, self.network_img_size, self.network_img_size).to(
                self.device).type_as(next(self.model.parameters())))  # run once
        self.K=[203.42144086256206, 0.0, 206.12517266886093, 
                0.0, 203.55319738398958, 146.4392791209304, 
                0.0, 0.0, 1.0]
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
        self.image_sub = message_filters.Subscriber(
            self.image_topic, Image)
        self.image_bub2 = message_filters.Subscriber(
            "/scan", LaserScan)
        self.image_bub = message_filters.Subscriber(
            "/camera2/image_raw", Image)
        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.image_bub2,self.image_bub],  1, 1)
        ts.registerCallback(self.image_cb)
        # Define publishers
        self.pub_ = rospy.Publisher(
            self.detected_objects_topic, BoundingBoxes, queue_size=10)
        self.pub_viz_ = rospy.Publisher(
            self.published_image_topic, Image, queue_size=10)
        rospy.loginfo("Launched node for object detection")

        # Spin
        rospy.spin()
    def lidar_cb(self,data):
        self.ranges=data

    def pubTf(self,position, orientation):
        """
        publish find object to tf2
        :param position:
        :param orientation:
        :return:
        """
        global t
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "/base_link"
        t.child_frame_id = "/object"
        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]
        quaternion = tf.transformations.quaternion_from_euler(orientation[0], orientation[1], orientation[2])

        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        tf2_br.sendTransform(t)

    def kalamnFilter(self, v):
        global intila
        global P
        global x_hat

        
        v=np.array(v)
        if intila==0:
            x_hat = v
            prev = v[0:1]
            intila=1
        else:
            
            K = P*self.C.transpose()*(self.C*P*self.C.transpose() + self.R).I
            j=np.multiply(x_hat,self.C)
            a=np.multiply(K.transpose(),np.matrix(v - j))
            x_hat = x_hat + a
            x_hat = x_hat*self.A.transpose()
            P = self.A*P*self.A.transpose() + self.Q
            P = (self.I - K*self.C)*P
            return x_hat

        return x_hat
    def image_cb(self, data,data2,data3):
        global prev
        global error_sum
        global n
        
        f = open('/home/nehalnevle/test3.csv', 'a')
        writer = csv.writer(f)
        self.ranges=data2.ranges
        try:
            self.cv_img = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(e)
        try:
            self.cv_img2 = self.bridge.imgmsg_to_cv2(data3, "rgb8")
        except CvBridgeError as e:
            print(e)
        # Initialize detection results
        B = 50
        
        detection_results = BoundingBoxes()
        detection_results.header = data.header
        detection_results.image_header = data.header
        
        input_img = self.preprocess(self.cv_img)
        input_img2 = self.preprocess(self.cv_img2)
        #input_img = Variable(input_img.type(torch.FloatTensor))
        #input_img2 = Variable(input_img2.type(torch.FloatTensor))

        # Get detections from network
        with torch.no_grad():
            input_img = torch.from_numpy(input_img).to(self.device,dtype=torch.half)
            input_img2 = torch.from_numpy(input_img2).to(self.device,dtype=torch.half)
            detections = self.model(input_img)[0]
            detections2 = self.model(input_img2)[0]
            detections = non_max_suppression(detections, self.conf_thres, self.iou_thres,
                                             classes=self.classes, agnostic=self.agnostic_nms)
            det1 = detections[0]
            detections2 = non_max_suppression(detections2, self.conf_thres, self.iou_thres,
                                             classes=self.classes, agnostic=self.agnostic_nms)
            det2=detections2[0]
            
        alpha = 127.22339616    
        # Parse detections
        i=0
        ct = CentroidTracker()
        j=1
        rects=[]
        
        if detections[0] is not None:
            bbox_xywh1 = xyxy2xywh(det1[:,0:4]).cpu()
            confs1 = det1[:, 4:5].cpu()
            bbox_xywh2 = xyxy2xywh(det2[:,0:4]).cpu()
            confs2 = det2[:, 4:5].cpu()
            try:
                outputs = self.deepsort.update(bbox_xywh1, confs1, self.cv_img)
                outputs2 = self.deepsort2.update(bbox_xywh2, confs2, self.cv_img2)
            
            
                bbox_xyxy2 = outputs2[:][0:4]
                
                bbox_xyxy = outputs[:][0:4]
            except:
                pass

            for i in range(len(detections[0])):
                kp=0.2
                try:
                    
                    xmin2, ymin2, xmax2, ymax2,w2=bbox_xyxy2[i]
                    
                except:
                    pass
                # Get xmin, ymin, xmax, ymax, confidence and class
                xmin, ymin, xmax, ymax, conf, det_class = detections[0][i]
                try:
                    xmin, ymin, xmax, ymax,w=bbox_xyxy[i]
                except:
                    pass
                if self.names[int(det_class)] not in ['bin_rect','bin_side','bin_cir']:
                    continue
                try:
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
                    pad_x2 = max(self.h - self.w, 0) * \
                        (self.network_img_size/max(self.h, self.w))
                    pad_y2 = max(self.w - self.h, 0) * \
                    (self.network_img_size/max(self.h, self.w))
                    unpad_h2 = self.network_img_size-pad_y2
                    unpad_w2 = self.network_img_size-pad_x2
                    rects.append([xmin_unpad, ymin_unpad, xmax_unpad, ymax_unpad])
                except:
                    pass
                
                try:
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
                    xmin_unpad = xmin_unpad
                    xmax_unpad = xmax_unpad
                    ymin_unpad =  ymin_unpad
                    ymax_unpad = ymax_unpad
                    pad_x2 = max(self.h - self.w, 0) * \
                        (self.network_img_size/max(self.h, self.w))
                    pad_y2 = max(self.w - self.h, 0) * \
                    (self.network_img_size/max(self.h, self.w))
                    unpad_h2 = self.network_img_size-pad_y2
                    unpad_w2 = self.network_img_size-pad_x2
                    rects.append([xmin_unpad, ymin_unpad, xmax_unpad, ymax_unpad])
                except:
                    pass
                try:
                   
                    xmin_unpad2 = ((xmin2-pad_x2//2)/unpad_w2)*self.w
                    xmax_unpad2 = ((xmax2-xmin2)/unpad_w2)*self.w + xmin_unpad2
                    ymin_unpad2 = ((ymin2-pad_y2//2)/unpad_h2)*self.h
                    ymax_unpad2 = ((ymax2-ymin2)/unpad_h2)*self.h + ymin_unpad2
                    xmin_unpad2 = xmin_unpad2.cpu().detach().numpy()
                    xmax_unpad2 = xmax_unpad2.cpu().detach().numpy()
                    ymin_unpad2 =  ymin_unpad2.cpu().detach().numpy()
                    ymax_unpad2 = ymax_unpad2.cpu().detach().numpy()
                    center_point_right=((xmin_unpad2-xmax_unpad2//2),(ymin_unpad2-ymax_unpad2//2))
                    center_point=((xmin_unpad-xmax_unpad//2),(ymin_unpad-ymax_unpad//2))
                    
                    frame_right, frame_left = calibration.undistortRectify(self.cv_img, self.cv_img2)
                    depth2 = tri.find_depth(center_point_right, center_point, frame_right, frame_left, B, f, alpha)
                    depth2=depth2/100
                    
                except Exception as e:
                    kp=0
                    depth2=0
                    pass
                try:
                    
                    xmin_unpad2 = ((xmin2-pad_x2//2)/unpad_w2)*self.w
                    xmax_unpad2 = ((xmax2-xmin2)/unpad_w2)*self.w + xmin_unpad2
                    ymin_unpad2 = ((ymin2-pad_y2//2)/unpad_h2)*self.h
                    ymax_unpad2 = ((ymax2-ymin2)/unpad_h2)*self.h + ymin_unpad2
                    xmin_unpad2 = xmin_unpad2
                    xmax_unpad2 = xmax_unpad2
                    ymin_unpad2 =  ymin_unpad2
                    ymax_unpad2 = ymax_unpad2
                    center_point_right=((xmin_unpad2-xmax_unpad2//2),(ymin_unpad2-ymax_unpad2//2))
                    center_point=((xmin_unpad-xmax_unpad//2),(ymin_unpad-ymax_unpad//2))
                    
                    frame_right, frame_left = calibration.undistortRectify(self.cv_img, self.cv_img2)
                    depth2 = tri.find_depth(center_point_right, center_point, frame_right, frame_left, B, f, alpha)
                
                    depth2=depth2/100
                    
                except Exception as e:
                    print(e)
                    kp=0
                    depth2=0
                    pass
                # Populate darknet message
                detection_msg = BoundingBox()
                detection_msg.xmin = int(xmin_unpad)
                detection_msg.xmax = int(xmax_unpad)
                detection_msg.ymin = int(ymin_unpad)
                detection_msg.ymax = int(ymax_unpad)
                detection_msg.probability = float(conf)
                detection_msg.Class = self.names[int(det_class)]
                
                angle1=(2.22/408)*(xmin_unpad)
                index1=int((angle1+0.462)//0.014032435603439808)
                angle2=(2.22/408)*(xmax_unpad)
                index2=int((angle2+0.462)//0.014032435603439808)
                depth1=statistics.mean(self.ranges[index1:index2])
                
                if depth2<0:
                    depth2=depth2*-1
                
                if depth1==nan:
                    depth1=0
                fx=203.42144086256206
                fy=203.55319738398958
                rotation_rad=[0,0,0]
                cx=206.12517266886093
                cy=146.4392791209304
                kp=0.2
                kn=1-kp
                try:
                    depth2=depth2.cpu().detach().numpy()
                except:
                    pass
                depth=kn*depth1+kp*depth2
                #depth=depth/2
                v=([depth,(depth) * (((410-(xmax_unpad-xmin_unpad//2))-cx)/fx),prev[0]-depth,prev[1]-(((410-(xmax_unpad-xmin_unpad//2))-cx)/fx)])
                print("diff;",prev[0]-depth)
                if abs(prev[0]-depth)>5:
                    depth=prev[0]
                prev=[depth,(depth) * (((410-(xmax_unpad-xmin_unpad//2))-cx)/fx)]
                v=([depth,(depth) * (((410-(xmax_unpad-xmin_unpad//2))-cx)/fx),prev[0]-depth,prev[1]-(((410-(xmax_unpad-xmin_unpad//2))-cx)/fx)])
                v_out=(v)
                strq=str(j)
                error=v_out[0]-depth1
                n+=1
                if error==float('nan'):
                    error=0
                print(u)
                _rate = 10.                 # this is a voracious application, so I recommend to lower the frequency, if it is not critical

                MIN_MATCH_COUNT = 10        # the lower the value, the more sensitive the filter
                blur_threshold = 300        # the higher the value, the more sensitive the filter
                max_dist = 10. 
                error_sum+=error**2
                rms=(error_sum/n)**0.5
                writer.writerow([depth1,depth2,depth,v_out[0]])
                print("lidar:",depth1," stereo",depth2," fused:",depth,"kalman out:",v_out[0],"rms:",rms)
                t = TransformStamped()
                tf2_br = tf2_ros.TransformBroadcaster()
                t.header.stamp = rospy.Time.now()
                t.header.frame_id = "/base_link"
                t.child_frame_id = "/object_"+strq
                j+=1
                size_image = 2.             # the width of the image in meters

                use_image = False           # uses a known image
                image_path = "image.jpg"    # path to known image

                show_image = True
                image_proc = image_processing.ImageEstimation(MIN_MATCH_COUNT, 300, use_image, size_image, image_path, show_image)
                image_proc.max_dist = max_dist
                frame, trans, rot = image_proc.update(self.cv_img, 0)
                orientation = rot
                t.transform.translation.x = v_out[0]
                t.transform.translation.y = v_out[1]
                t.transform.translation.z = 0
                quaternion = tf.transformations.quaternion_from_euler(orientation[0], orientation[1], orientation[2])

                t.transform.rotation.x = quaternion[0]
                t.transform.rotation.y = quaternion[1]
                t.transform.rotation.z = quaternion[2]
                t.transform.rotation.w = quaternion[3]
                tf2_br.sendTransform(t)
                # Append in overall detection message
                detection_results.bounding_boxes.append(detection_msg)
        objects = ct.update(rects)
        
        # Publish detection results
        self.pub_.publish(detection_results)
        f.close()
        # Visualize detection results
        if (self.publish_image):
            self.visualize_and_publish(detection_results,objects, self.cv_img)
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

    def visualize_and_publish(self, output,objects, imgIn):
        global u
        # Copy image and visualize
        imgOut = imgIn.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        thickness = 2
        
        #for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            
        #    text = "ID {}".format(objectID)
        #    cv2.putText(imgOut, text, (centroid[0] - 10, centroid[1] - 10),
        #        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #    cv2.circle(imgOut, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
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
        if u<12:
            print("done")
            cv2.imwrite("/home/nehalnevle/test"+str(u)+".png",imgOut)
        u+=0.2
        # Publish visualization image
        image_msg = self.bridge.cv2_to_imgmsg(imgOut, "rgb8")
        image_msg.header.frame_id = 'camera'
        image_msg.header.stamp = rospy.Time.now()
        self.pub_viz_.publish(image_msg)


if __name__ == '__main__':
    rospy.init_node('detector')
    f = open("/home/nehalnevle/test3.csv", 'w')
    # Convert the image to OpenCV
    writer = csv.writer(f)
    header = ['lidar', 'stereo', 'fused', 'kalman']
    writer.writerow(header)
    f.close()
    # Define detector object
    dm = Detector()
