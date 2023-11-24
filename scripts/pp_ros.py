#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
import time
import numpy as np
from pyquaternion import Quaternion
import numpy as np
import torch
import time
import sys

sys.path.append("/home/rui/pp_ros/src/mmdetection3d_ROS/scripts/mmdet3d")
from mmdet3d.apis import inference_detector, init_model
from mmdet3d.core.points import get_points_type

class Pointpillars_ROS:

    def lidar_callback(self, msg):

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

        # 在回调函数中，读取雷达点云，转换成numpy数组
        points = np.array(pc2.read_points_list(msg))
         # print('points:', points)
        # print('points_shape:', points.shape)
        # points = points[:, :4]  # 只取x,y,z,i，因为ouster读取的数据中不止这四个值
        # print('points:', points)
        # print('points_shape:', points.shape)

        points_class = get_points_type('LIDAR')  # 使用mmdet3d中的函数定义数据类型
        points_mmdet3d = points_class(points, points_dim=points.shape[-1], attribute_dims=None)  # 将数据转换成mmdet3d的格式
        torch.cuda.synchronize()  # 多卡同步
        start_time = time.perf_counter()  # 记录推理开始时间
        # pcd = '/home/rui/mmdet3d_ws/src/mmdetection3d_ROS/tools/checkpoint/kitti_000008.bin'
        result, data = inference_detector(model, points_mmdet3d)  # 开始推理
                # print('\nresult:', result, '\ndata:', data)
        # print('\ndata_type=', type(data))
        # print('\ndata:points=', data['points'])
        # print('\ndata:points_shape=', type(data['points']))

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time  # 计算单帧推理所花费时间
        print(f'single frame: {elapsed*1000:.1f} ms')
        # 推理阶段的NMS
        scores = result[0]['scores_3d'].numpy()
        mask = scores > 0.5
        scores = scores[mask]
        boxes_lidar = result[0]['boxes_3d'][mask].tensor.cpu().numpy()
        label = result[0]['labels_3d'][mask].numpy()
        # 创建BoundingBoxArray消息，存储多个目标的bbox信息
        arr_bbox = BoundingBoxArray()
        for i in range(boxes_lidar.shape[0]):
            # 创建BoundingBox消息，存储单个目标的bbox信息
            bbox = BoundingBox()
            # 将包围盒 bbox 的帧ID（frame ID）设置为与接收到的消息 msg 的帧ID相同
            bbox.header.frame_id = msg.header.frame_id
            # 将包围盒 bbox 的时间戳（timestamp）设置为当前时间。时间戳用于标识消息的时间信息，以便其他节点可以根据时间戳对消息进行排序和处理
            bbox.header.stamp = rospy.Time.now()
            bbox.pose.position.x = float(boxes_lidar[i][0])
            bbox.pose.position.y = float(boxes_lidar[i][1])
            bbox.pose.position.z = float(boxes_lidar[i][2])  # + float(boxes_lidar[5]) / 2
            bbox.dimensions.x = float(boxes_lidar[i][3])  # width
            bbox.dimensions.y = float(boxes_lidar[i][4])  # length
            bbox.dimensions.z = float(boxes_lidar[i][5])  # height
            # 用四元数来表示bbox在空间中的朝向信息
            q = Quaternion(axis=(0, 0, 1), radians=float(boxes_lidar[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            bbox.label = label[i]
            # print(label[i])
            arr_bbox.boxes.append(bbox)
        arr_bbox.header.frame_id = msg.header.frame_id
        # 若存在bbox则发布bbox话题
        if len(arr_bbox.boxes) != 0:
            pub_bbox.publish(arr_bbox)
            self.publish_test(points, msg.header.frame_id)

    def publish_test(self, cloud, frame_id):
        '''
        将点云数据转换为 ROS 消息类型，并发布到名为 pub_velo 的话题中
        '''
        header = Header()
        header.stamp = rospy.Time()  # 设置时间戳为当前时间
        header.frame_id = "/base_link"  # 设置坐标系 ID 为 "/base_link"，通常是机器人的基座坐标系
        # header.frame_id = frame_id
        # 包含点云数据字段信息的列表，每个字段包含名称、偏移量、数据类型和计数信息
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]  # ,PointField('label', 16, PointField.FLOAT32, 1)
        # creat_cloud不像read，他必须要有fields,而且field定义有两种。一个是上面的，一个是下面的fields=_make_point_field(4)
        # 创建一个 PointCloud2 类型的 ROS 消息 msg_segment，使用点云数据 cloud、头信息 header 和数据字段 fields
        msg_segment = pc2.create_cloud(header=header, fields=fields, points=cloud)
        pub_velo.publish(msg_segment)
        # pub_image.publish(image_msg)


        # def detector_server(req):
#     # 解析提交的数据
#     if (req.object_pose):
#         # object_x = 
#         # object_y = 
#         # object_z =
#         # object_w =
#         # object_l =
#         # object_h =
#         # object_r =
#         resp = detectorResponse()
#         return resp


if __name__ == '__main__':
    
    # 1.导入检测模型
    # 填写自己的实际路径
    checkpoint_file = '/home/rui/pp_ros/src/mmdetection3d_ROS/checkpoints/pp3class.pth'
    config_file = '/home/rui/pp_ros/src/mmdetection3d_ROS/scripts/pp_kitti-3d-3class.py'
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    
    # 2.实例化类，该类的作用是接收雷达数据并检测然后绘制检测框
    global sec
    sec = Pointpillars_ROS()

    # 3.初始化节点
    rospy.init_node('pointpillars_ros_node', anonymous=True)

    # subscriber订阅雷达数据，可以是rosbag也可以是实时数据，修改节点名称即可
    # 回调函数完成目标检测和bbox绘制
    sub_velo = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, sec.lidar_callback, queue_size=1,
                                buff_size=2 ** 12)

    # publisher发布检测结果detections
    pub_velo = rospy.Publisher("/modified", PointCloud2, queue_size=1)
    pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=10)

    # # 6.创建服务对象
    # server = rospy.Service('detector', detector, detector_server)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        print("Shutting down")
