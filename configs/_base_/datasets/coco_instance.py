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
'''
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
'''
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
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
evaluation = dict(metric=['bbox', 'segm'])
