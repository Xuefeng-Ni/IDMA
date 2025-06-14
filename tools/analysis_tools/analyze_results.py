# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.evaluation import eval_map
from mmdet.core.visualization import imshow_gt_det_bboxes
from mmdet.datasets import build_dataset, get_loading_pipeline


def bbox_map_eval(det_result, annotation):
    """Evaluate mAP of single image det result.

    Args:
        det_result (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotation (dict): Ground truth annotations where keys of
             annotations are:

            - bboxes: numpy array of shape (n, 4)
            - labels: numpy array of shape (n, )
            - bboxes_ignore (optional): numpy array of shape (k, 4)
            - labels_ignore (optional): numpy array of shape (k, )

    Returns:
        float: mAP
    """

    # use only bbox det result
    if isinstance(det_result, tuple):
        bbox_det_result = [det_result[0]]
    else:
        bbox_det_result = [det_result]
    # mAP
    iou_thrs = np.linspace(
        .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    mean_aps = []
    for thr in iou_thrs:
        mean_ap, _ = eval_map(
            bbox_det_result, [annotation], iou_thr=thr, logger='silent')
        mean_aps.append(mean_ap)
    return sum(mean_aps) / len(mean_aps)


class ResultVisualizer:
    """Display and save evaluation results.

    Args:
        show (bool): Whether to show the image. Default: True
        wait_time (float): Value of waitKey param. Default: 0.
        score_thr (float): Minimum score of bboxes to be shown.
           Default: 0
    """

    def __init__(self, show=False, wait_time=0, score_thr=0):
        self.show = show
        self.wait_time = wait_time
        self.score_thr = score_thr

    def _save_image_gts_results(self, dataset, results, gts, eval_mode, mAPs, score_thr_cls, out_dir=None):
        mmcv.mkdir_or_exist(out_dir)

        for mAP_info in mAPs:
            index, mAP = mAP_info
            data_info = dataset.prepare_train_img(index)

            num_class = 1   #3

            #iof&iog
            if eval_mode == 'iof':
                data_id = data_info['img_info']['id']
                if data_id in gts:
                    bboxes = np.zeros_like(data_info['gt_bboxes'])
                    labels = np.zeros_like(data_info['gt_labels'])
                    iogs = np.zeros_like(labels,dtype=np.float)
                    bboxes[:gts[data_id][0].shape[0], :] = gts[data_id][0][:, :-1]
                    if num_class == 3:
                        bboxes[gts[data_id][0].shape[0]:(gts[data_id][0].shape[0]+gts[data_id][1].shape[0]), :] = gts[data_id][1][:, :-1]
                        bboxes[(gts[data_id][0].shape[0]+gts[data_id][1].shape[0]):, :] = gts[data_id][2][:, :-1]
                    labels[:gts[data_id][0].shape[0]] = 0
                    if num_class == 3:
                        labels[gts[data_id][0].shape[0]:(gts[data_id][0].shape[0]+gts[data_id][1].shape[0])] = 1
                        labels[(gts[data_id][0].shape[0]+gts[data_id][1].shape[0]):] = 2
                    iogs[:gts[data_id][0].shape[0]] = gts[data_id][0][:, -1]
                    if num_class == 3:
                        iogs[gts[data_id][0].shape[0]:(gts[data_id][0].shape[0]+gts[data_id][1].shape[0])] = gts[data_id][1][:, -1]
                        iogs[(gts[data_id][0].shape[0]+gts[data_id][1].shape[0]):] = gts[data_id][2][:, -1]
                    data_info['gt_bboxes'] = bboxes
                    data_info['gt_labels'] = labels
                    data_info['gt_iogs'] = iogs
                else:
                    data_info['gt_iogs'] = np.zeros_like(data_info['gt_labels'], dtype=np.float)
            else:
                data_id = data_info['img_info']['id']
                if data_id in gts:
                    bboxes = np.zeros_like(data_info['gt_bboxes'])
                    labels = np.zeros_like(data_info['gt_labels'])
                    ious = np.zeros_like(labels,dtype=np.float)
                    bboxes[:gts[data_id][0].shape[0], :] = gts[data_id][0][:, :-1]
                    if num_class == 3:
                        bboxes[gts[data_id][0].shape[0]:(gts[data_id][0].shape[0]+gts[data_id][1].shape[0]), :] = gts[data_id][1][:, :-1]
                        bboxes[(gts[data_id][0].shape[0]+gts[data_id][1].shape[0]):, :] = gts[data_id][2][:, :-1]
                    labels[:gts[data_id][0].shape[0]] = 0
                    if num_class == 3:
                        labels[gts[data_id][0].shape[0]:(gts[data_id][0].shape[0]+gts[data_id][1].shape[0])] = 1
                        labels[(gts[data_id][0].shape[0]+gts[data_id][1].shape[0]):] = 2
                    ious[:gts[data_id][0].shape[0]] = gts[data_id][0][:, -1]
                    if num_class == 3:
                        ious[gts[data_id][0].shape[0]:(gts[data_id][0].shape[0]+gts[data_id][1].shape[0])] = gts[data_id][1][:, -1]
                        ious[(gts[data_id][0].shape[0]+gts[data_id][1].shape[0]):] = gts[data_id][2][:, -1]
                    data_info['gt_bboxes'] = bboxes
                    data_info['gt_labels'] = labels
                    data_info['gt_ious'] = ious
                else:
                    data_info['gt_ious'] = np.zeros_like(data_info['gt_labels'], dtype=np.float)

            # calc save file path
            filename = data_info['filename']
            if data_info['img_prefix'] is not None:
                filename = osp.join(data_info['img_prefix'], filename)
            else:
                filename = data_info['filename']
            fname, name = osp.splitext(osp.basename(filename))
            save_filename = fname + '_' + str(round(mAP, 3)) + name
            out_file = osp.join(out_dir, save_filename)
            imshow_gt_det_bboxes(
                data_info['img'],
                data_info,
                results[index],
                eval_mode,
                dataset.CLASSES,
                gt_bbox_color= [(0, 255, 0), (60, 186, 75), (81, 204, 33)], #dataset.PALETTE,
                gt_text_color=(200, 200, 200),
                gt_mask_color= [(0, 255, 0), (60, 186, 75), (81, 204, 33)],  #dataset.PALETTE,
                det_bbox_color=dataset.PALETTE,
                det_text_color=(200, 200, 200),
                det_mask_color=dataset.PALETTE,
                thickness=4,
                show=self.show,
                score_thr=self.score_thr,
                score_thr_cls=score_thr_cls,
                wait_time=self.wait_time,
                out_file=out_file)

    def evaluate_and_show(self,
                          dataset,
                          results,
                          gts,
                          eval_mode,
                          score_thr_cls,
                          topk=20,
                          show_dir='work_dir',
                          eval_fn=None):
        """Evaluate and show results.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Det results from test results pkl file
            topk (int): Number of the highest topk and
                lowest topk after evaluation index sorting. Default: 20
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
            eval_fn (callable, optional): Eval function, Default: None
        """

        assert topk > 0
        if (topk * 2) > len(dataset):
            topk = len(dataset) // 2 + 1

        if eval_fn is None:
            eval_fn = bbox_map_eval
        else:
            assert callable(eval_fn)

        prog_bar = mmcv.ProgressBar(len(results))
        _mAPs = {}
        for i, (result, ) in enumerate(zip(results)):
            # self.dataset[i] should not call directly
            # because there is a risk of mismatch
            data_info = dataset.prepare_train_img(i)

            mAP = eval_fn(result, data_info['ann_info'])
            _mAPs[i] = mAP
            prog_bar.update()

        # descending select topk image
        _mAPs = list(sorted(_mAPs.items(), key=lambda kv: kv[1]))
        good_mAPs = _mAPs[-topk:]   #[::-1]
        bad_mAPs = _mAPs[:topk]     #[::-1]

        good_dir = osp.abspath(osp.join(show_dir, 'good'))
        bad_dir = osp.abspath(osp.join(show_dir, 'bad'))

        self._save_image_gts_results(dataset, results, gts, eval_mode, good_mAPs, score_thr_cls, good_dir)
        self._save_image_gts_results(dataset, results, gts, eval_mode, bad_mAPs, score_thr_cls, bad_dir)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        'show_dir', help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=0,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--topk',
        default=20,
        type=int,
        help='saved Number of the highest topk '
        'and lowest topk after index sorting')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0,
        help='score threshold (default: 0.)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    root_dir = '/mnt/sda/nxf/IDMA/work_dirs/tood_x101_fpn_1x_coco_rsdds/'
    score_thr_cls = np.array([0.68])
    eval_mode = 'iof'
    if eval_mode == 'iou':
        args.show_dir = root_dir + 'output_images_iou'
    elif eval_mode == 'iof':
        args.show_dir = root_dir + 'output_images_iof_iog_new'
    else:
        args.show_dir = root_dir + 'output_images_gt'
    args.prediction_path = root_dir + 'results_eval_iou_det.pkl'
    prediction_iou_gt_path = root_dir + 'results_eval_iou_gt.pkl'
    prediction_iof_path = root_dir + 'results_eval_iof.pkl'
    prediction_iog_path = root_dir + 'results_eval_iog.pkl'

    args.show_score_thr = 0.3
    args.show = False
    args.topk = 2846 #688

    mmcv.check_file_exist(args.prediction_path)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    cfg.data.test.pop('samples_per_gpu', 0)
    cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.prediction_path)
    outputs_iou_gt = mmcv.load(prediction_iou_gt_path)
    outputs_iof = mmcv.load(prediction_iof_path)
    outputs_iog = mmcv.load(prediction_iog_path)

    result_visualizer = ResultVisualizer(args.show, args.wait_time,
                                         args.show_score_thr)
    if eval_mode == 'iou' or eval_mode == 'gt':
        result_visualizer.evaluate_and_show(
            dataset, outputs, outputs_iou_gt, eval_mode, score_thr_cls, topk=args.topk, show_dir=args.show_dir)
    else:
        result_visualizer.evaluate_and_show(
            dataset, outputs_iof, outputs_iog, eval_mode, score_thr_cls, topk=args.topk, show_dir=args.show_dir)


if __name__ == '__main__':
    main()
