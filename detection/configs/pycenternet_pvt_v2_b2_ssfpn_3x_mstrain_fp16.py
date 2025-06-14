_base_ = [
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='PyCenterNetDetector',    #TOODFOVEA_SPPNet
#    pretrained='/home/nxf/IDMA/work_dirs/atss_pvt_v2_b2_fpn_3x_mstrain_fp16/epoch_29.pth',  #pretrained/pvt_v2_b2.pth
    pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth', #pvt_v2_b2
#    pretrained=None,
    backbone=dict(
        type='pvt_v2_b2',
        style='pytorch'),
    neck=dict(
        type='SSFPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=1,
        num_outs=5,
        add_extra_convs='on_input'
    ),
    bbox_head=dict(
        type='PyCenterNetHead',
        num_classes=3,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        shared_stacked_convs=1,
        first_kernel_size=3,
        kernel_size=1,
        corner_dim=64,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        norm_cfg=norm_cfg,
        conv_module_style='dcn',  # norm or dcn, norm is faster
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='GIoULoss', loss_weight=1.0),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0),
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            loss_weight=0.25),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_sem=dict(
            type='SEPFocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.1)),
    # training and testing settings
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))
# augmentation strategy originates from DETR / Sparse RCNN
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[[
             dict(type='Resize',
                  img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                             (736, 1333), (768, 1333), (800, 1333)],
                  multiscale_mode='value',
                  keep_ratio=True)],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
data = dict(train=dict(pipeline=train_pipeline))

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])

# do not use apex fp16
# runner = dict(type='EpochBasedRunner', max_epochs=36)

# use apex fp16
'''
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
'''
runner = dict(type='EpochBasedRunner', max_epochs=36)
find_unused_parameters = True

'''
dict(type='AutoAugment',
     policies=[[
         dict(type='Resize',
              img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                         (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                         (736, 1333), (768, 1333), (800, 1333)],
              multiscale_mode='value',
              keep_ratio=True)],
         [
             dict(type='Resize',
                  img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                  multiscale_mode='value',
                  keep_ratio=True),
             dict(type='RandomCrop',
                  crop_type='absolute_range',
                  crop_size=(384, 600),
                  allow_negative_crop=True),
             dict(type='Resize',
                  img_scale=[(480, 1333), (512, 1333), (544, 1333),
                             (576, 1333), (608, 1333), (640, 1333),
                             (672, 1333), (704, 1333), (736, 1333),
                             (768, 1333), (800, 1333)],
                  multiscale_mode='value',
                  override=True,
                  keep_ratio=True)]
     ]),
'''