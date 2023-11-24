#!/usr/bin/env python3

from mmdet3d.apis import inference_detector, init_model

config_file = 'src/mmdetection3d_ROS/date_checkpoint_cfg/pp_kitti-3d-3class.py'
checkpoint_file = 'src/mmdetection3d_ROS/date_checkpoint_cfg/pp3class.pth'

# 从配置文件和预训练的模型文件中构建模型
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 测试单个文件并可视化结果
point_cloud = 'src/mmdetection3d_ROS/date_checkpoint_cfg/kitti_000008.bin'
result, data = inference_detector(model, point_cloud)
print('result:', result, '\ndata:', data)
# 可视化结果并且将结果保存到 'results' 文件夹
model.show_results(data, result, out_dir='results')
