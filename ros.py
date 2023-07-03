#!/usr/bin/env python3

import rospy

from sensor_msgs.msg import PointCloud2, PointField

import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

import time
import numpy as np
from pyquaternion import Quaternion

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import scipy.linalg as linalg
import time
import sys
import os

sys.path.append("/home/d/code/mmdetection3d_ros/mmdet/src/mmdet")

from argparse import ArgumentParser
import math
from mmdet3d.apis import inference_detector, init_model
from mmdet3d.core.points import get_points_type
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker,MarkerArray

class Pointpillars_ROS:

    def lidar_callback(self, msg):

        # build the model from a config file and a checkpoint file
        # 填写自己的实际路径：
        # config_file = '/home/d/mmdet/src/mmdet/tools/configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py'
        # checkpoint_file = '/home/d/mmdet/src/mmdet/tools/checkpoint/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth'
        checkpoint_file = '/home/d/code/mmdetection3d_ros/mmdet/src/mmdet/tools/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
        config_file = '/home/d/code/mmdetection3d_ros/mmdet/src/mmdet/tools/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
        model = init_model(config_file, checkpoint_file, device='cuda:0')

        #pcl_msg = pc2.read_points(msg, field_names = ("x", "y", "z", "intensity"), skip_nans=True)
        # 这里的field_names可以不要，不要就是把数据全部读取进来。也可以用field_names = ("x", "y", "z")这个只读xyz坐标
        # 得到的pcl_msg是一个generator生成器，如果要一次获得全部的点，需要转成list
        #np_p = np.array(list(pcl_msg), dtype=np.float32)
        # print(np_p.shape)
        # 旋转轴
        # rand_axis = [0,1,0]
        # 旋转角度
        # yaw = 0.1047
        # yaw = 0.0
        # 返回旋转矩阵
        # rot_matrix = self.rotate_mat(rand_axis, yaw)
        # np_p_rot = np.dot(rot_matrix, np_p[:,:3].T).T

        # convert to xyzi point cloud
        # x = np_p[:, 0].reshape(-1)
        # print(np.max(x),np.min(x))
        # y = np_p[:, 1].reshape(-1)
        # z = np_p[:, 2].reshape(-1)
        # if np_p.shape[1] == 4: # if intensity field exists
        #    i = np_p[:, 3].reshape(-1)
        # else:
        #   i = np.zeros((np_p.shape[0], 1)).reshape(-1)
        # points = np.stack((x, y, z, i)).T

        # 在回调函数中
        points = np.array(pc2.read_points_list(msg))
        points_class = get_points_type('LIDAR')
        points_mmdet3d = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        # points_mmdet3d = points_class(points, points_dim=5, attribute_dims=None)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        result, data = inference_detector(model, points_mmdet3d)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        print(f'single frame: {elapsed*1000:.1f} ms')
        scores = result[0]['scores_3d'].numpy()
        mask = scores > 0.5
        scores = scores[mask]
        boxes_lidar = result[0]['boxes_3d'][mask].tensor.cpu().numpy()
        label = result[0]['labels_3d'][mask].numpy()
        # boxes_lidar=result[0]['boxes_3d'].tensor.cpu().numpy()
        # label=result[0]['labels_3d'].numpy()
        # scores = result[0]['scores_3d'].numpy()
        #print(points.size)
        #print(data)
        #print(points[0])
        data = []
        points1 = []
        cats=["Pedestrian", "Cyclist", "Car"]
        cat = ["m"]
        arr_bbox = BoundingBoxArray()
        markerArray = MarkerArray()
        for i in range(scores.size):
            bbox = BoundingBox()
            marker = Marker()
            marker.header = msg.header
            marker.type = marker.TEXT_VIEW_FACING
            marker.id = int(label[i])
            marker.text = f"{int(boxes_lidar[i][0])}{cat[0]} {cats[int(label[i])]}"
            marker.action = marker.ADD
            marker.frame_locked = True
            marker.lifetime = rospy.Duration(1)
            marker.scale.x, marker.scale.y,marker.scale.z = 0.8, 0.8, 0.8
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0, 0, 1.0
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = boxes_lidar[i][0], boxes_lidar[i][1], boxes_lidar[i][2]+1
            # boxes_lidar=result[i]['boxes_3d'].tensor.cpu().numpy()
            # label=result[i]['labels_3d'].numpy()
            # scores = result[i]['scores_3d'].numpy()

            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2])  # + float(boxes_lidar[5]) / 2
            bbox.dimensions.x = float(boxes_lidar[i][3])  # width
            bbox.dimensions.y = float(boxes_lidar[i][4])  # length
            bbox.dimensions.z = float(boxes_lidar[i][5])  # height
            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            # bbox.value = 0.6
            bbox.label = label[i]
            # print(label[i])
            arr_bbox.boxes.append(bbox)
            markerArray.markers.append(marker)
            data.append(boxes_lidar[i][0])
            data.append(boxes_lidar[i][1])
            data.append(boxes_lidar[i][2])
            data.append(boxes_lidar[i][3])
            data.append(boxes_lidar[i][4])
            data.append(boxes_lidar[i][5])
            data.append(boxes_lidar[i][6])
        #print(np.array(data).size)
        #print(data)
        #start_time1 = time.perf_counter()
        #for j in range((int) (points.size/4)):
          #  for k in range(scores.size):
                #if (self.check_point_in_box(points[j], data[k + 0], data[k + 1], data[k + 2],
                   #                    data[k + 3], data[k + 4], data[k + 5], data[k + 6])):
                 #if (self.check_point_in_box(points[j], boxes_lidar[k][0], boxes_lidar[k][1], boxes_lidar[k][2],
                             #       boxes_lidar[k][3], boxes_lidar[k][4], boxes_lidar[k][5], boxes_lidar[k][6]) == True):
                    #break
                 #else:
                   # if (k == scores.size-1):
                      #print(points[j])

                      #points1.append(points[j])
        #elapsed1 = time.perf_counter() - start_time
        #print(f'pushtime: {elapsed1 * 1000:.1f} ms')
        #print(points1.size)
        arr_bbox.header.frame_id = msg.header.frame_id
        # arr_bbox.header.stamp = rospy.Time.now()
        left_top = Float64MultiArray(data=data)
        if len(arr_bbox.boxes) is not 0:
            pub_bbox.publish(arr_bbox)
            marker_pub.publish(markerArray)
            pub_bboxtest.publish(left_top)
            self.publish_test(points, msg.header.frame_id)

    def check_point_in_box(self,pts, box0, box1, box2, box3, box4, box5, box6):
        """
        pts[x,y,z]
        box[c_x,c_y,c_z,dx,dy,dz,heading]
    """

        shift_x = pts[0] - box0
        shift_y = pts[1] - box1
        shift_z = pts[2] - box2
        cos_a = math.cos(box6)
        sin_a = math.sin(box6)
        dx, dy, dz = box3, box4, box5
        local_x = shift_x * cos_a + shift_y * sin_a;
        local_y = shift_y * cos_a - shift_x * sin_a;
        if ((abs(shift_z) > dz / 2.0) | (abs(local_x) > dx / 2.0) | (abs(local_y) > dy / 2.0)):
            return False
        return True

    def publish_test(self, cloud, frame_id):
        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = "/base_link"
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]  # ,PointField('label', 16, PointField.FLOAT32, 1)
        # creat_cloud不像read，他必须要有fields,而且field定义有两种。一个是上面的，一个是下面的fields=_make_point_field(4)
        msg_segment = pc2.create_cloud(header=header, fields=fields, points=cloud)

        pub_velo.publish(msg_segment)
        # pub_image.publish(image_msg)


def _make_point_field(num_field):
    msg_pf1 = pc2.PointField()
    msg_pf1.name = np.str_('x')
    msg_pf1.offset = np.uint32(0)
    msg_pf1.datatype = np.uint8(7)
    msg_pf1.count = np.uint32(1)

    msg_pf2 = pc2.PointField()
    msg_pf2.name = np.str_('y')
    msg_pf2.offset = np.uint32(4)
    msg_pf2.datatype = np.uint8(7)
    msg_pf2.count = np.uint32(1)

    msg_pf3 = pc2.PointField()
    msg_pf3.name = np.str_('z')
    msg_pf3.offset = np.uint32(8)
    msg_pf3.datatype = np.uint8(7)
    msg_pf3.count = np.uint32(1)

    msg_pf4 = pc2.PointField()
    msg_pf4.name = np.str_('intensity')
    msg_pf4.offset = np.uint32(16)
    msg_pf4.datatype = np.uint8(7)
    msg_pf4.count = np.uint32(1)

    if num_field == 4:
        return [msg_pf1, msg_pf2, msg_pf3, msg_pf4]

    msg_pf5 = pc2.PointField()
    msg_pf5.name = np.str_('label')
    msg_pf5.offset = np.uint32(20)
    msg_pf5.datatype = np.uint8(4)
    msg_pf5.count = np.uint32(1)

    return [msg_pf1, msg_pf2, msg_pf3, msg_pf4, msg_pf5]


if __name__ == '__main__':
    global sec
    sec = Pointpillars_ROS()

    rospy.init_node('pointpillars_ros_node', anonymous=True)

    # subscriber

    sub_velo = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, sec.lidar_callback, queue_size=1,
                                buff_size=2 ** 12)

    # publisher
    pub_velo = rospy.Publisher("/modified", PointCloud2, queue_size=1)
    pub_bboxtest = rospy.Publisher("/detections_track", Float64MultiArray, queue_size=10)
    # pub_bboxtest = rospy.Publisher("/detections_track", detection, queue_size=10)

    pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=10)
    marker_pub = rospy.Publisher("/detections11", MarkerArray, queue_size=10)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        print("Shutting down")
