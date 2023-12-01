#!/usr/bin/env python3

import rospy
from mmdetection3d_ROS.srv import detector, detectorRequest, detectorResponse


if __name__ == "__main__":

    # 1.初始化 ROS 节点
    rospy.init_node("client_test")
    # 2.创建请求对象
    client = rospy.ServiceProxy("detector", detector)
    # 请求前，等待服务已经就绪
    client.wait_for_service()
    # 3.发送请求,接收并处理响应
    req = detectorRequest()
    req.success = bool(True)
    # 4.进行服务调用
    resp = client.call(req)
    rospy.loginfo("响应结果:%d",resp.object_pose)