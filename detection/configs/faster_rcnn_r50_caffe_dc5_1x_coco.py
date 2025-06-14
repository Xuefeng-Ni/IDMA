_base_ = [
    '../configs/_base_/models/faster_rcnn_r50_caffe_dc5.py',
    '../configs/_base_/datasets/coco_detection.py',
    '../configs/_base_/schedules/schedule_2x.py', '../configs/_base_/default_runtime.py'
]
'''
model = dict(
    pretrained='/home/ni/Desktop/nxf/IDMA/pretrained/vgg16_caffe-292e1171.pth',
    backbone=dict(
        type='VGG',
        depth=16,
        with_bn=False,
        num_stages=5,
        dilations=(1, 1, 1, 1, 1),
        out_indices=(3, ),   #(1, 2, 3, 4),
        frozen_stages=-1,
        bn_eval=False,
        bn_frozen=False,
        ceil_mode=False,
        with_last_pool=False,
##        pretrained=None #'open-mmlab://vgg16_caffe'
    )
)
'''