_base_ = './faster_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='/home/nxf/.cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth')))   #torchvision://resnet101
