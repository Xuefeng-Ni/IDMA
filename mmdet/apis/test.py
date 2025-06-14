# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results

import numpy as np
from skimage.io import imread

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    results_dict = {}
    results2 = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]

        results.extend(result)

        #test
        dataset = 'SDDMTR'  # 'RSDDS'
        if dataset == 'SDDMTR':
            num_classes = 3
            conf_thred = [0.0, 0.0, 0.0]    # Please set yor own threshold
        else:
            num_classes = 1
            conf_thred = [0.0]  # Please set yor own threshold

        if num_classes == 1:
            result_dict = {0: result[0][0]}
        elif num_classes == 3:
            result_dict = {0: result[0][0], 1: result[0][1], 2: result[0][2]}

        if num_classes == 2:
            result_filter = {0: [], 1: []}
            conf_box = {0: [], 1: []}
        elif num_classes == 1:
            result_filter = {0: []}
            conf_box = {0: []}
        elif num_classes == 3:
            result_filter = {0: [], 1: [], 2: []}
            conf_box = {0: [], 1: [], 2: []}
        conf_box_length = 0
        for j in range(0, num_classes):
            for i in range(0, result_dict[j].shape[0]):
                if result_dict[j][i, 4] < conf_thred[j]:
                    conf_box[j].append(i)
                    conf_box_length = conf_box_length + 1
        for j in range(0, num_classes):
            conf_box[j] = np.array(conf_box[j])
            if conf_box[j].size != 0:
                result_filter[j] = np.delete(result_dict[j], conf_box[j], axis=0)
            else:
                result_filter[j] = result_dict[j]
        results_dict.update({data['img_metas'][0].data[0][0]['id']: result_filter})

        #area filter
        area_thred = [36, 64, 100]
##        area_thred = [0, 0, 0]
##        area_thred = [100]
        if num_classes == 2:
            area_filter = {0: [], 1: []}
            small_box = {0: [], 1: []}
        elif num_classes == 1:
            area_filter = {0: []}
            small_box = {0: []}
        elif num_classes == 3:
            area_filter = {0: [], 1: [], 2: []}
            small_box = {0: [], 1: [], 2: []}
        small_box_length = 0
        for j in range(0, num_classes):
            for i in range(0, result_filter[j].shape[0]):
                if ((result_filter[j][i, 3] - result_filter[j][i, 1]) * (result_filter[j][i, 2] - result_filter[j][i, 0])) <= area_thred[j]:
                    small_box[j].append(i)
                    small_box_length = small_box_length + 1
        for j in range(0, num_classes):
            small_box[j] = np.array(small_box[j])
            if small_box[j].size != 0:
                area_filter[j] = np.delete(result_filter[j], small_box[j], axis=0)
            else:
                area_filter[j] = result_filter[j]
        results_dict.update({data['img_metas'][0].data[0][0]['id']: area_filter})
##        results2.extend([list(area_filter.values())])

        '''
        #rail region filter
        file_name = data['img_metas'][0].data[0][0]['ori_filename']
##        seg_path = '/mnt/sda/nxf/IDMA/data/defects/SegmentationClassContext_new/' + file_name[:-4] + '.png'
        seg_path = '/mnt/sda/nxf/IDMA/data/defects/SegmentationClassContext_new_ciassn/' + file_name[:-4] + '.png'
        seg_img = imread(seg_path)
        if len(seg_img.shape) == 3:
            seg = seg_img[:, :, 2]
        else:
            seg = seg_img
        if num_classes == 2:
            seg_filter = {0: [], 1: []}
            seg_box = {0: [], 1: []}
        elif num_classes == 1:
            seg_filter = {0: []}
            seg_box = {0: []}
        elif num_classes == 3:
            seg_filter = {0: [], 1: [], 2: []}
            seg_box = {0: [], 1: [], 2: []}
        seg_box_length = 0
        for j in range(0, num_classes):
            for i in range(0, area_filter[j].shape[0]):
                area_sum = (np.round(area_filter[j][i, 2]) - np.round(area_filter[j][i, 0])) * (np.round(area_filter[j][i, 3]) - np.round(area_filter[j][i, 1]))
                seg_region = seg[int(np.round(area_filter[j][i, 1])):int(np.round(area_filter[j][i, 3])), int(np.round(area_filter[j][i, 0])):int(np.round(area_filter[j][i, 2]))]
                area_seg = float(np.sum(seg_region))
                area_pro = area_seg / area_sum
                if area_pro < 0.3:
                    seg_box[j].append(i)
                    seg_box_length = seg_box_length + 1
        for j in range(0, num_classes):
            seg_box[j] = np.array(seg_box[j])
            if seg_box[j].size != 0:
                seg_filter[j] = np.delete(area_filter[j], seg_box[j], axis=0)
            else:
                seg_filter[j] = area_filter[j]
        results_dict.update({data['img_metas'][0].data[0][0]['id']: seg_filter})
        results2.extend([list(seg_filter.values())])
        '''
        #without filter
##        results_dict.update({data['img_metas'][0].data[0][0]['id']: result_dict})

        for _ in range(batch_size):
            prog_bar.update()
    return results, results_dict, results2


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
