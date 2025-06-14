_base_ = './faster_rcnn_r50_fpn_2x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='/home/ni/.cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth')))
