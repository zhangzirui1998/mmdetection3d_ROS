#!/usr/bin/env python3
 
import rospy
 
from sensor_msgs.msg import PointCloud2,PointField
 
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
 
import sys
sys.path.append("/home/d/mmdet/src/mmdet")
 
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.core.points import get_points_type
 
 
class Pointpillars_ROS:
 
    def lidar_callback(self, msg):
        
        # build the model from a config file and a checkpoint file
        #填写自己的实际路径：
        #config_file = '/home/d/mmdet/src/mmdet/tools/configs/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.py'
        #checkpoint_file = '/home/d/mmdet/src/mmdet/tools/checkpoint/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth'
        checkpoint_file = '/home/d/mmdet/src/mmdet/tools/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth'
        config_file = '/home/d/mmdet/src/mmdet/tools/configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
        model = init_model(config_file, checkpoint_file, device='cuda:0')
       

        #pcl_msg = pc2.read_points(msg, field_names = ("x", "y", "z", "intensity"), skip_nans=True)
         #这里的field_names可以不要，不要就是把数据全部读取进来。也可以用field_names = ("x", "y", "z")这个只读xyz坐标
        #得到的pcl_msg是一个generator生成器，如果要一次获得全部的点，需要转成list
        #np_p = np.array(list(pcl_msg), dtype=np.float32)
        #print(np_p.shape)
        # 旋转轴
        #rand_axis = [0,1,0]
        #旋转角度
        #yaw = 0.1047
        #yaw = 0.0
        #返回旋转矩阵
        #rot_matrix = self.rotate_mat(rand_axis, yaw)
        #np_p_rot = np.dot(rot_matrix, np_p[:,:3].T).T
 
        # convert to xyzi point cloud
        #x = np_p[:, 0].reshape(-1)
        #print(np.max(x),np.min(x))
        #y = np_p[:, 1].reshape(-1)
        #z = np_p[:, 2].reshape(-1)
        #if np_p.shape[1] == 4: # if intensity field exists
        #    i = np_p[:, 3].reshape(-1)
        #else:
         #   i = np.zeros((np_p.shape[0], 1)).reshape(-1)
        #points = np.stack((x, y, z, i)).T

        #在回调函数中 
        points=np.array(pc2.read_points_list(msg))
        points_class = get_points_type('LIDAR')
        points_mmdet3d = points_class(points, points_dim=points.shape[-1], attribute_dims=None)
        #points_mmdet3d = points_class(points, points_dim=5, attribute_dims=None)
        result, data = inference_detector(model, points_mmdet3d)
        scores = result[0]['scores_3d'].numpy()
        mask = scores>0.5
        scores = scores[mask]
        boxes_lidar=result[0]['boxes_3d'][mask].tensor.cpu().numpy()
        label=result[0]['labels_3d'][mask].numpy()
        #boxes_lidar=result[0]['boxes_3d'].tensor.cpu().numpy()
        #label=result[0]['labels_3d'].numpy()
        #scores = result[0]['scores_3d'].numpy()
        
 
        arr_bbox = BoundingBoxArray()
        for i in range(scores.size):
            bbox = BoundingBox()

            #boxes_lidar=result[i]['boxes_3d'].tensor.cpu().numpy()
            #label=result[i]['labels_3d'].numpy()
            #scores = result[i]['scores_3d'].numpy()

            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2]) #+ float(boxes_lidar[5]) / 2
            bbox.dimensions.x = float(boxes_lidar[i][3])  # width
            bbox.dimensions.y = float(boxes_lidar[i][4])  # length
            bbox.dimensions.z = float(boxes_lidar[i][5])  # height
            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            #bbox.value = 0.6 
            bbox.label = label[i]
            #print(label[i])
            arr_bbox.boxes.append(bbox)
 
        arr_bbox.header.frame_id = msg.header.frame_id
        #arr_bbox.header.stamp = rospy.Time.now()
 
        if len(arr_bbox.boxes) is not 0:
            pub_bbox.publish(arr_bbox)
            self.publish_test(points, msg.header.frame_id)
 
    def publish_test(self, cloud, frame_id):
        header = Header()
        header.stamp = rospy.Time()
        header.frame_id = frame_id
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]  # ,PointField('label', 16, PointField.FLOAT32, 1)
        #creat_cloud不像read，他必须要有fields,而且field定义有两种。一个是上面的，一个是下面的fields=_make_point_field(4)
        msg_segment = pc2.create_cloud(header = header,fields=fields,points = cloud)
 
        pub_velo.publish(msg_segment)
        #pub_image.publish(image_msg)
 
 
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
    pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=10)
 
    try:
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        print("Shutting down")
