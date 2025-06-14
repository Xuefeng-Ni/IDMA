# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

import os
import pvt
import pvt_v2


def parse_args():
    parser = ArgumentParser()
#    parser.add_argument('img', help='Image file')
#    parser.add_argument('config', help='Config file')
#    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):

    args.config = '/mnt/sda/nxf/IDMA/detection/configs/tood_pvt_v2_b2_fpn_3x_fp16.py'
    args.checkpoint = '/mnt/sda/nxf/IDMA/work_dirs/main_results/seed_tood_down_up_att_cat_fusion_step_cat_pvt_v2_b2_fpn_bfp_att2_0-5_skip_pvt_2_1_1_0_3x_fp16_casiou_gfocal_centerness_per_class_loss_iof_iog/epoch_28.pth'
    images_dir = '/mnt/sda/nxf/IDMA/data/defects/images/test'

#    args.config = '/mnt/sda/nxf/IDMA/configs/yolact/yolact_r50_1x8_coco.py'
#    args.config = '/mnt/sda/nxf/IDMA/configs/mask_rcnn/mask_rcnn_r101_fpn_2x_coco.py'
#    args.checkpoint = '/mnt/sda/nxf/IDMA/work_dirs/yolact_r50_1x8_coco_no_pretrain/epoch_42.pth'
#    images_dir = '/mnt/sda/nxf/IDMA/data/rail_instance/images/test2017'

    demo_im_names = os.listdir(images_dir)  #[::-1]
    if not os.path.exists('/mnt/sda/nxf/IDMA/output_images/'):
        os.makedirs('/mnt/sda/nxf/IDMA/output_images/')

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    for im_name in demo_im_names:
        img = os.path.join(images_dir, im_name)
        # test a single image
        result = inference_detector(model, img)  #args.img
        # show the results
        show_result_pyplot(
            model,
            img,   #args.img
            result,
            im_name,
            palette=args.palette,
            score_thr=args.score_thr)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
