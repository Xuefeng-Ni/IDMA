_base_ = [
    '../configs/_base_/models/faster_rcnn_r50_caffe_c4_ori.py',
    '../configs/_base_/datasets/coco_detection.py',
    '../configs/_base_/schedules/schedule_2x.py', '../configs/_base_/default_runtime.py'
]
model = dict(
    pretrained=None,
#    backbone=dict(
#        depth=101,
#        init_cfg=dict(type='Pretrained',
#                      checkpoint='/home/ni/.cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth'))
#    neck=dict(
#        in_channels=[128, 256, 512, 512])
)
