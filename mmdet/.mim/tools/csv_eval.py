from __future__ import print_function

import numpy as np
import json
import os

import torch
import codecs
from itertools import chain

import cv2


def compute_overlap(a, b, detected_annotations):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    union_a = intersection / ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))
    union_a[np.where(union_a >= 0.5)] = 1.0

    interbox_idx = np.where((intersection > 0))
    interbox = []
    if (interbox_idx[1].size != 0) & (a.shape[1] == 6):
        assigned_annotation = np.argmax(intersection / union_a, axis=1)
        max_overlap = union_a[0, assigned_annotation]
        for i in range(len(interbox_idx[1])):
            if max_overlap < 1:
                interbox.append([max(a[0, 0], b[interbox_idx[1][i], 0]), max(a[0, 1], b[interbox_idx[1][i], 1]),
                                 min(a[0, 2], b[interbox_idx[1][i], 2]), min(a[0, 3], b[interbox_idx[1][i], 3]),
                                 area[interbox_idx[1][i]]])

    if a.shape[1] == 5:
        union_a = intersection / a[:, 4]
        np.fill_diagonal(union_a, 0)
        union_a = np.triu(union_a)# - np.eye(intersection.shape[0], dtype=float)

    return intersection / ua, union_a, intersection / area, interbox


def compute_overlap2(a, b, detected_annotations):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    union_a = intersection / ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))

    interbox_idx = np.where((intersection > 0))
    interbox = []

    if (interbox_idx[1].size != 0) & (a.shape[1] == 4):
        assigned_annotation = np.argmax(union_a, axis=1)
        max_overlap = (union_a)[0, assigned_annotation]
        for i in range(len(interbox_idx[1])):
            if max_overlap < 0.5:
                interbox.append([max(a[0, 0], b[interbox_idx[1][i], 0]), max(a[0, 1], b[interbox_idx[1][i], 1]),
                                min(a[0, 2], b[interbox_idx[1][i], 2]), min(a[0, 3], b[interbox_idx[1][i], 3]),
                                (a[0, 2] - a[0, 0]) * (a[0, 3] - a[0, 1])])

    if a.shape[1] == 5:
        union_a = intersection / a[:, 4]
        np.fill_diagonal(union_a, 0)
        union_a = np.triu(union_a)# - np.eye(intersection.shape[0], dtype=float)

    return intersection / ua, union_a, intersection / area, interbox

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(dataset, retinanet, score_threshold=0.05, max_detections=100, save_path=None):
    """ Get the detections from the retinanet using the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]
    # Arguments
        dataset         : The generator used to run images through the retinanet.
        retinanet           : The retinanet to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(dataset.num_classes())] for j in range(len(dataset))]

    retinanet.eval()
    
    with torch.no_grad():

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']

            # run network
            scores, labels, boxes = retinanet(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes  = boxes.cpu().numpy()

            # correct boxes for image scale
            boxes /= scale

            # select indices which have a score above the threshold
            indices = np.where(scores > score_threshold)[0]
            if indices.shape[0] > 0:
                # select those scores
                scores = scores[indices]

                # find the order with which to sort the scores
                scores_sort = np.argsort(-scores)[:max_detections]

                # select detections
                image_boxes      = boxes[indices[scores_sort], :]
                image_scores     = scores[scores_sort]
                image_labels     = labels[indices[scores_sort]]
                image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = image_detections[image_detections[:, -1] == label, :-1]
            else:
                # copy detections to all_detections
                for label in range(dataset.num_classes()):
                    all_detections[index][label] = np.zeros((0, 5))

            print('{}/{}'.format(index + 1, len(dataset)), end='\r')

    return all_detections


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.
    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = annotations[num_detections, 5]
    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(len(generator))]

    for i in range(len(generator)):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

        print('{}/{}'.format(i + 1, len(generator)), end='\r')

    return all_annotations


def evaluate(
    generator,
    all_detections,
    coco_names,
    iou_threshold=0.5,
    score_threshold=0.05,
    max_detections=100,
    save_path=None
):
    """ Evaluate a given dataset using a given retinanet.
    # Arguments
        generator       : The generator that represents the dataset to evaluate.
        retinanet           : The retinanet to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        score_threshold : The score confidence threshold to use for detections.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """

    # gather all detections and annotations

    eval_mode = 'area-iof'   #'iou' 'iof' 'mm-iof' 'area-iof'
    dataset = 'SDDMTR' # 'RSDDS'
    if dataset == 'SDDMTR':
        num_classes = 3
    else:
        num_classes = 1

    all_annotations={}
    eval_root_dir = '/mnt/sda/nxf/IDMA/eval_txt'

    for key, value in generator.items():
        if num_classes == 2:
            add_item = {key: {0: [], 1: []}}
        elif num_classes == 3:
            add_item = {key: {0: [], 1: [], 2: []}}
        elif num_classes == 1:
            add_item = {key: {0: []}}
        elif num_classes == 6:
            add_item = {key: {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}}
        all_annotations.update(add_item)
        for j, v in enumerate(value):
            if v['category_id'] == 0:
                all_annotations[key][0].append(v['bbox'])
            elif v['category_id'] == 1:
                all_annotations[key][1].append(v['bbox'])
            else:
                all_annotations[key][2].append(v['bbox'])
        all_annotations[key][0] = np.array(all_annotations[key][0])
        if num_classes == 2:
            all_annotations[key][1] = np.array(all_annotations[key][1])
        elif num_classes == 3:
            for j in range(1, 3):
                all_annotations[key][j] = np.array(all_annotations[key][j])
        elif num_classes == 6:
            for j in range(1, 6):
                all_annotations[key][j] = np.array(all_annotations[key][j])

        for i in range(all_annotations[key][0].shape[0]):
            if all_annotations[key][0].size != 0:
                all_annotations[key][0][i][2] += all_annotations[key][0][i][0]
                all_annotations[key][0][i][3] += all_annotations[key][0][i][1]
        if num_classes == 2:
            for i in range(all_annotations[key][1].shape[0]):
                if all_annotations[key][1].size != 0:
                    all_annotations[key][1][i][2] += all_annotations[key][1][i][0]
                    all_annotations[key][1][i][3] += all_annotations[key][1][i][1]
        elif num_classes == 3:
            for j in range(1, 3):
                for i in range(all_annotations[key][j].shape[0]):
                    if all_annotations[key][j].size != 0:
                        all_annotations[key][j][i][2] += all_annotations[key][j][i][0]
                        all_annotations[key][j][i][3] += all_annotations[key][j][i][1]
        elif num_classes == 6:
            for j in range(1, 6):
                for i in range(all_annotations[key][j].shape[0]):
                    if all_annotations[key][j].size != 0:
                        all_annotations[key][j][i][2] += all_annotations[key][j][i][0]
                        all_annotations[key][j][i][3] += all_annotations[key][j][i][1]

    average_precisions = {}
    if num_classes == 1:
        sum_annotations = [0.0]
        sum_TP = [0.0]
        sum_FP = [0.0]
        sum_FN = [0.0]
        sum_TP_PR = [0.0]
        sum_FP_PR = [0.0]
    elif num_classes == 2:
        sum_annotations = [0.0, 0.0]
        sum_TP = [0.0, 0.0]
        sum_FP = [0.0, 0.0]
        sum_FN = [0.0, 0.0]
    elif num_classes == 3:
        sum_annotations = [0.0, 0.0, 0.0]
        sum_TP = [0.0, 0.0, 0.0]
        sum_FP = [0.0, 0.0, 0.0]
        sum_TP_PR = [0.0, 0.0, 0.0]
        sum_FP_PR = [0.0, 0.0, 0.0]
        sum_FN = [0.0, 0.0, 0.0]

    if not os.path.exists(eval_root_dir):
        os.makedirs(eval_root_dir)
    for label in range(num_classes):  #generator.num_classes()
        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        false_positives_pr = np.zeros((0,))
        true_positives_pr  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        false_negatives = 0.0

        for i in (all_annotations.keys() & all_detections.keys()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []
            interboxes = []
            interboxes_pr = []
            cover_annotations = np.zeros((1, annotations.shape[0]))

            true_positives_img = np.zeros((0,))
            false_positives_img = np.zeros((0,))
            num_annotations_img = annotations.shape[0]
            detections_eval = np.zeros((np.size(detections, 0), 7))
            annotations_eval = np.zeros((np.size(annotations, 0), 5))

            bbox_idx = 0
            for d in detections:
                detections_eval[bbox_idx][:-1] = d
                scores = np.append(scores, d[4])

                if annotations.shape[0] == 0:
                    false_positives_pr = np.append(false_positives_pr, 1)
                    true_positives_pr  = np.append(true_positives_pr, 0)
                    detections_eval[bbox_idx][-1] = 0.0
                    bbox_idx += 1
                    continue

                overlaps, union_a, union_b, interbox_pr = compute_overlap(np.expand_dims(d, axis=0), annotations, detected_annotations)
                interboxes_pr.append(interbox_pr)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap           = overlaps[0, assigned_annotation]
                max_union_a           = union_a[0, assigned_annotation]
                max_union_b           = union_b[0, assigned_annotation]

                if eval_mode == 'iou':
                    condition = (max_overlap >= iou_threshold and assigned_annotation not in detected_annotations)
                elif eval_mode == 'iof':
                    condition = (max_union_a >= iou_threshold and assigned_annotation not in detected_annotations)
                else:
                    condition = (max_union_a >= iou_threshold)
                if condition:
                    if eval_mode == 'area-iof':
                        false_positives_pr = np.append(false_positives_pr, 0)
                        true_positives_pr  = np.append(true_positives_pr, 1)
                        detections_eval[bbox_idx][-1] = 1.0
                    else:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)

                else:
                    if len(interbox_pr):
                        pred_area = (d * 10.0 ** 2).round()
                        intersum_area = np.zeros(
                            (pred_area[2].astype(int) - pred_area[0].astype(int), pred_area[3].astype(int) - pred_area[1].astype(int)))
                        interbox_area = (np.array(interbox_pr) * 10.0 ** 2).round()
                        interbox_area[:, 0] = interbox_area[:, 0] - pred_area[0]
                        interbox_area[:, 1] = interbox_area[:, 1] - pred_area[1]
                        interbox_area[:, 2] = interbox_area[:, 2] - pred_area[0]
                        interbox_area[:, 3] = interbox_area[:, 3] - pred_area[1]
                        for j in range(interbox_area.shape[0]):
                            intersum_area[interbox_area[j, 0].astype(int):(interbox_area[j, 2] + 1).astype(int),
                            interbox_area[j, 1].astype(int):(interbox_area[j, 3] + 1).astype(int)] = 1
                        iofs_sum = np.sum(intersum_area) / (intersum_area.shape[0] * intersum_area.shape[1])
                    else:
                        iofs_sum = 0.0

                    if eval_mode == 'area-iof':
                        false_positives_pr = np.append(false_positives_pr, 1-iofs_sum)    #np.sum(union_b)
                        true_positives_pr  = np.append(true_positives_pr, iofs_sum)
                        detections_eval[bbox_idx][-1] = iofs_sum
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

                bbox_idx += 1
            all_detections[i][label] = detections_eval

            bbox_idx = 0
            cover_annotations = np.zeros((1, detections.shape[0]))
            overlaps_sum = np.zeros((annotations.shape[0], detections.shape[0]))
            for d in annotations:
                annotations_eval[bbox_idx][:-1] = d

                if detections.shape[0] == 0:
                    annotations_eval[bbox_idx][-1] = 0.0
                    bbox_idx += 1
                    continue

                overlaps, union_a, union_b, interbox = compute_overlap2(np.expand_dims(d, axis=0), detections[:, :-1],
                                                                       detected_annotations)
                overlaps_sum[bbox_idx, :] = overlaps
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]
                max_union_a = union_a[0, assigned_annotation]
                max_union_b = union_b[0, assigned_annotation]
                if max_union_a < 0.5:
                    interboxes.append(interbox)

                cover_idx = np.where(union_a >= 0.5, 1, 0)
                cover_annotations = cover_annotations + cover_idx

                if eval_mode == 'iou':
                    condition = (max_overlap >= iou_threshold and assigned_annotation not in detected_annotations)
                elif eval_mode == 'iof':
                    condition = (max_union_a >= iou_threshold and assigned_annotation not in detected_annotations)
                else:
                    condition = (max_union_a >= iou_threshold)
                if condition:
                    false_positives = np.append(false_positives, 0)
                    true_positives = np.append(true_positives, 1)

                    detected_annotations.append(assigned_annotation)
                    false_positives_img = np.append(false_positives_img, 0)
                    true_positives_img = np.append(true_positives_img, 1)
                    annotations_eval[bbox_idx][-1] = 1.0
                else:  # if max_overlap < iou_threshold:
                    if len(interbox):
                        gt_area = (d * 10.0 ** 2).round()
                        intersum_area = np.zeros(
                            (gt_area[2].astype(int) - gt_area[0].astype(int), gt_area[3].astype(int) - gt_area[1].astype(int)))
                        interbox_area = (np.array(interbox) * 10.0 ** 2).round()
                        interbox_area[:, 0] = interbox_area[:, 0] - gt_area[0]
                        interbox_area[:, 1] = interbox_area[:, 1] - gt_area[1]
                        interbox_area[:, 2] = interbox_area[:, 2] - gt_area[0]
                        interbox_area[:, 3] = interbox_area[:, 3] - gt_area[1]
                        for j in range(interbox_area.shape[0]):
                            intersum_area[interbox_area[j, 0].astype(int):(interbox_area[j, 2] + 1).astype(int),
                            interbox_area[j, 1].astype(int):(interbox_area[j, 3] + 1).astype(int)] = 1
                        iogs_sum = np.sum(intersum_area) / (intersum_area.shape[0] * intersum_area.shape[1])
                    else:
                        iogs_sum = 0.0

                    if eval_mode == 'area-iof':
                        false_positives = np.append(false_positives, 1 - iogs_sum)  # np.sum(union_b)
                        true_positives = np.append(true_positives, iogs_sum)
                        annotations_eval[bbox_idx][-1] = iogs_sum
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

                    false_positives_img = np.append(false_positives_img, 1)
                    true_positives_img = np.append(true_positives_img, 0)
                bbox_idx += 1
            all_annotations[i][label] = annotations_eval

            false_positives_add = np.sum(np.where(np.sum(overlaps_sum, axis=0), 0, 1))
            for f in range(false_positives_add):
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)

            FN_img = np.sum(cover_annotations.flatten().astype(int)==0) #single-multiple union>0.5
            false_negatives = false_negatives + FN_img

            # compute false positives and true positives
            false_positives_img = np.cumsum(false_positives_img)
            true_positives_img = np.cumsum(true_positives_img)
            if true_positives_img.size != 0:
                recall_img = true_positives_img / num_annotations_img
                precision_img = true_positives_img / np.maximum(true_positives_img + false_positives_img, np.finfo(np.float64).eps)
                F_measure_img = 2 * recall_img[-1] * precision_img[-1] / (recall_img[-1] + precision_img[-1])
                out_file = os.path.join(eval_root_dir, coco_names[i]['file_name'][:-4] + '.txt')  # [:im_name.rindex('.')]
                with codecs.open(out_file, 'w', 'utf-8') as f:
                    f.write('{:.2f} {:.2f} {:.2f}\n'.format(recall_img[-1], precision_img[-1], F_measure_img))
            else:
                out_file = os.path.join(eval_root_dir, coco_names[i]['file_name'][:-4] + '.txt')  # [:im_name.rindex('.')]
                with codecs.open(out_file, 'w', 'utf-8') as f:
                    f.write('{:.2f} {:.2f} {:.2f}\n'.format(0, 0, 0))

        for i in (all_annotations.keys() ^ all_detections.keys()):
            true_positives_img = np.zeros((0,))
            false_positives_img = np.zeros((0,))
            num_annotations_img = annotations.shape[0]

            if i in all_annotations.keys():
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                annotations_eval = np.zeros((np.size(annotations, 0), 5))
                bbox_idx = 0

                for d in annotations:
                    annotations_eval[bbox_idx][:-1] = d
                    true_positives = np.append(true_positives, 0)
                    false_positives = np.append(false_positives, 0)
                    true_positives_pr = np.append(true_positives_pr, 0)
                    false_positives_pr = np.append(false_positives_pr, 0)
                    true_positives_img = np.append(true_positives_img, 0)
                    false_positives_img = np.append(false_positives_img, 0)
                    annotations_eval[bbox_idx][-1] = 0.0
                    bbox_idx += 1
                all_annotations[i][label] = annotations_eval
            else:
                detections = all_detections[i][label]
                detections_eval = np.zeros((np.size(detections, 0), 7))
                bbox_idx = 0
                for d in detections:
                    detections_eval[bbox_idx][:-1] = d
                    detections_eval[bbox_idx][-1] = 0.0
                    false_positives = np.append(false_positives, 1)
                    true_positives = np.append(true_positives, 0)
                    false_positives_pr = np.append(false_positives_pr, 1)
                    true_positives_pr = np.append(true_positives_pr, 0)
                    false_positives_img = np.append(false_positives_img, 1)
                    true_positives_img = np.append(true_positives_img, 0)
                    bbox_idx += 1
                all_detections[i][label] = detections_eval

            if true_positives_img.size != 0:
                recall_img = true_positives_img / num_annotations_img
                precision_img = true_positives_img / np.maximum(true_positives_img + false_positives_img, np.finfo(np.float64).eps)
                F_measure_img = 2 * recall_img[-1] * precision_img[-1] / (recall_img[-1] + precision_img[-1])
                out_file = os.path.join(eval_root_dir, coco_names[i]['file_name'][:-4] + '.txt')  # [:im_name.rindex('.')]
                with codecs.open(out_file, 'w', 'utf-8') as f:
                    f.write('{:.2f} {:.2f} {:.2f}\n'.format(recall_img[-1], precision_img[-1], F_measure_img))
            else:
                out_file = os.path.join(eval_root_dir, coco_names[i]['file_name'][:-4] + '.txt')  # [:im_name.rindex('.')]
                with codecs.open(out_file, 'w', 'utf-8') as f:
                    f.write('{:.2f} {:.2f} {:.2f}\n'.format(0, 0, 0))

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)
        false_positives_pr = np.cumsum(false_positives_pr)
        true_positives_pr  = np.cumsum(true_positives_pr)

        # compute recall and precision
        if eval_mode == 'mm-iof':
            recall = (num_annotations - false_negatives) / num_annotations   #single-multiple union>0.5
        else:
            recall = true_positives / num_annotations
        if eval_mode == 'area-iof':
            precision = true_positives_pr / np.maximum(true_positives_pr + false_positives_pr, np.finfo(np.float64).eps)
        else:
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
        if eval_mode == 'mm-iof':
            if precision.size == 0:
                F_measure = 0
            else:
                F_measure = 2 * recall * precision[-1] / np.maximum((recall + precision[-1]), np.finfo(np.float64).eps)  #single-multiple union>0.5
        else:
            if precision.size == 0:
                F_measure = 0
            else:
                F_measure = 2 * recall[-1] * precision[-1] / np.maximum((recall[-1] + precision[-1]), np.finfo(np.float64).eps)

        if label == 0:
            if eval_mode == 'mm-iof':
                print('{}: {}'.format('spalling_recall', recall))   #single-multiple union>0.5
            else:
                if recall.size == 0:
                    print('{}: {}'.format('spalling_recall', 0))
                else:
                    print('{}: {}'.format('spalling_recall', recall[-1]))
            if precision.size == 0:
                print('{}: {}'.format('spalling_precision', 0))
            else:
                print('{}: {}'.format('spalling_precision', precision[-1]))
            print('{}: {}'.format('spalling_F_measure', F_measure))
        elif label == 1:
            if eval_mode == 'mm-iof':
                print('{}: {}'.format('cracks_recall', recall))
            else:
                if recall.size == 0:
                    print('{}: {}'.format('cracks_recall', 0))
                else:
                    print('{}: {}'.format('cracks_recall', recall[-1]))
            if precision.size == 0:
                print('{}: {}'.format('cracks_precision', 0))
            else:
                print('{}: {}'.format('cracks_precision', precision[-1]))
            print('{}: {}'.format('cracks_F_measure', F_measure))
        else:
            if eval_mode == 'mm-iof':
                print('{}: {}'.format('squats_recall', recall))
            else:
                if recall.size == 0:
                    print('{}: {}'.format('squats_recall', 0))
                else:
                    print('{}: {}'.format('squats_recall', recall[-1]))
            if precision.size == 0:
                print('{}: {}'.format('squats_precision', 0))
            else:
                print('{}: {}'.format('squats_precision', precision[-1]))
            print('{}: {}'.format('squats_F_measure', F_measure))

        # compute average precision
        sum_annotations[label] = num_annotations
        if true_positives.size == 0:
            sum_TP[label] = 0
        else:
            sum_TP[label] = true_positives[-1]
            sum_TP_PR[label] = true_positives_pr[-1]
        if false_positives.size == 0:
            sum_FP[label] = 0
        else:
            sum_FP[label] = false_positives[-1]
            sum_FP_PR[label] = false_positives_pr[-1]
        sum_FN[label] = false_negatives

    N = 0.0
    TP = 0.0
    FP = 0.0
    TP_PR = 0.0
    FP_PR = 0.0
    FN = 0.0
    print('\nmAP:')
    for label in range(num_classes):
        N += sum_annotations[label]
        TP += sum_TP[label]
        FP += sum_FP[label]
        TP_PR += sum_TP_PR[label]
        FP_PR += sum_FP_PR[label]
        FN += sum_FN[label]
    if eval_mode == 'area-iof':
        PR = TP_PR / np.maximum((TP_PR + FP_PR), np.finfo(np.float64).eps)
    else:
        PR = TP / np.maximum((TP + FP), np.finfo(np.float64).eps)
    if eval_mode == 'mm-iof':
        RC = (N - FN) / N   ##single-multiple union>0.5
    else:
        RC = TP / N
    F = 2 * PR * RC / np.maximum((PR + RC), np.finfo(np.float64).eps)
    print('\n')
    print('{}: {}'.format('sum_recall', RC))
    print('{}: {}'.format('sum_precisions', PR))
    print('{}: {}'.format('sum_F-measure', F))

    outputs = []
    for k, v in all_detections.items():
        if num_classes == 3:
            outputs.append([v[0], v[1], v[2]])
        elif num_classes == 1:
            outputs.append([v[0]])
    
    return all_detections, outputs, all_annotations

