_base_ = '../_base_/default_runtime.py'

# model settings
img_size = 512
model = dict(
    type='YOLACT',
    backbone=dict(
        type='Res2Net',
        depth=101,
        num_stages=4,
        scales=4,
        base_width=26,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,  # do not freeze stem
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,  # update the statistics of bn
        zero_init_residual=False,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='/mnt/sda/nxf/IDMA/pretrained/res2net101_v1d_26w_4s_mmdetv2-f0a600f9.pth')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        upsample_cfg=dict(mode='bilinear')),
    bbox_head=dict(
        type='YOLACTHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=3,
            scales_per_octave=1,
            base_sizes=[8, 16, 32, 64, 128],
            ratios=[0.5, 1.0, 2.0],
            strides=[550.0 / x for x in [69, 35, 18, 9, 5]],
            centers=[(550 * 0.5 / x, 550 * 0.5 / x)
                     for x in [69, 35, 18, 9, 5]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            reduction='none',
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.5),
        num_head_convs=1,
        num_protos=32,
        use_ohem=True),
    mask_head=dict(
        type='YOLACTProtonet',
        in_channels=256,
        num_protos=32,
        num_classes=1,
        max_masks_to_train=100,
        loss_mask_weight=6.125),
    segm_head=dict(
        type='YOLACTSegmHead',
        num_classes=1,
        in_channels=256,
        loss_segm=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        # smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        iou_thr=0.5,
        top_k=200,
        max_per_img=100))
# dataset settings
dataset_type = 'RailDataset'
data_root = '/mnt/sda/nxf/IDMA/data/rail_instance/'
img_norm_cfg = dict(
    mean=[112.63, 112.63, 112.63], std=[64.44, 64.44, 64.44], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file='/mnt/sda/nxf/IDMA/data/rail_instance/annotations/instances_train2017.json',
        seg_file=
        '/mnt/sda/nxf/rail_seg/mmsegmentation-master/data/VOCdevkit/VOC2010_5/SegmentationClassContext/train/',
        img_prefix=data_root + 'images/train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file='/mnt/sda/nxf/IDMA/data/rail_instance/annotations/instances_val2017.json',
        seg_file=
        '/mnt/sda/nxf/rail_seg/mmsegmentation-master/data/VOCdevkit/VOC2010_5/SegmentationClassContext/test2/',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/mnt/sda/nxf/IDMA/data/rail_instance/annotations/instances_val2017.json',
        seg_file=
        '/mnt/sda/nxf/rail_seg/mmsegmentation-master/data/VOCdevkit/VOC2010_5/SegmentationClassContext/test2/',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=1e-5, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[20, 42, 49, 52])
runner = dict(type='EpochBasedRunner', max_epochs=55)
cudnn_benchmark = True
evaluation = dict(metric=['bbox', 'segm'])
