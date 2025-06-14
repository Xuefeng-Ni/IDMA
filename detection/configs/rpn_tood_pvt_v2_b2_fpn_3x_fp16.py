_base_ = [
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
num_stages = 6
num_proposals = 300
model = dict(
    type='SparseRCNN',
    # pretrained='pretrained/pvt_v2_b2.pth',
    pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth',
    backbone=dict(
        type='pvt_v2_b2',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='TOODHead',
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
                    #            strides=[8, 16, 32, 64, 128]),
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
                ##        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),  #bbox refine
                ##        loss_bbox_last=dict(type='GIoULoss', loss_weight=2.)),  #bbox refine
                loss_wh=dict(type='CASIoULoss', loss_weight=2.0),
                loss_wh_last=dict(type='CASIoULoss', loss_weight=2.0)) for _ in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                initial_epoch=4,
                initial_assigner=dict(type='ATSSAssigner', topk=9),
                assigner=dict(type='TaskAlignedAssigner', topk=13),
                sampler=dict(type='PseudoSampler'),
                alpha=1,
                beta=6,
                allowed_border=-1,
                pos_weight=-1,
                debug=False) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.000025 / 1.4, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
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
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(pipeline=train_pipeline))
lr_config = dict(policy='step', step=[27, 33])
runner = dict(type='EpochBasedRunner', max_epochs=36)
