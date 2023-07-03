# pointpillars_ros
A ros implement for pointpillars (mmdeection3d based)

## Usage
```python
catkin_make
source devel/setup.bash
roslaunch xxx pointpillars.launch
```
This repository is based on mmdetection3d for ros adaptation, please install mmdetection3d first. 

This makes it a complete ros package to run.

It is worth saying that mm3d_ros is based on OpenPCDet_ros rewritten, thank you very much for the open source code!!!
### TODO LIST:
    Multi Object Tracking

```python
@misc{mmdet3d2020,
    title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},
    author={MMDetection3D Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
    year={2020}
}
```
