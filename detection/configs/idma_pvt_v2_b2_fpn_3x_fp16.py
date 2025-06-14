_base_ = [
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='IDMA',
    pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth', #pvt_v2_b2
    backbone=dict(
        type='pvt_v2_b2',
        style='pytorch'),
    neck=[dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=6),
        dict(
        type='BFP',
        in_channels=256,
        num_levels=6,
        refine_level=2,
        refine_type='non_local')],
    bbox_head=dict(
        type='IDMAHead',
        num_classes=3,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, 8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.4,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='CASIoULoss', loss_weight=2.0, mode='iou'),
        loss_bbox_last=dict(type='CASIoULoss', loss_weight=2.0),
        loss_scale=dict(type='L1Loss', loss_weight=0.1),
        loss_iof=dict(type='IoULoss', loss_weight=1.5, mode='linear', iou_mode='iof'),
        loss_iog=dict(type='IoULoss', loss_weight=1.5, mode='linear', iou_mode='iog'),
        loss_wh=dict(type='CASIoULoss', loss_weight=2.0),
        loss_wh_last=dict(type='CASIoULoss', loss_weight=2.0)),
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

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[27, 33])

# do not use apex fp16
# runner = dict(type='EpochBasedRunner', max_epochs=36)

# use apex fp16

runner = dict(type='EpochBasedRunner', max_epochs=36)
find_unused_parameters = True
custom_hooks = [dict(type='SetEpochInfoHook')]
