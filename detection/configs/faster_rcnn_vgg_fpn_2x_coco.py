_base_ = './faster_rcnn_r50_fpn_2x_coco.py'
model = dict(
    pretrained=None,
    backbone=dict(
        type='VGG',
        depth=16,
        with_bn=False,
        num_stages=5,
        dilations=(1, 1, 1, 1, 1),
        out_indices=(1, 2, 3, 4),   #(1, 2, 3, 4),
        frozen_stages=-1,
        bn_eval=False,
        bn_frozen=False,
        ceil_mode=False,
        with_last_pool=False,
        pretrained='open-mmlab://vgg16_caffe'
    ),
    neck=dict(
        in_channels=[128, 256, 512, 512])
)
