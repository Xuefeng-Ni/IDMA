# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import mmcv
import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..builder import LOSSES
from .utils import weighted_loss
import math
import numpy as np


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def iou_loss(pred, target, linear=False, mode='log', iou_mode='iof', gt_bbox=None, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).

    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn('DeprecationWarning: Setting "linear=True" in '
                      'iou_loss is deprecated, please use "mode=`linear`" '
                      'instead.')
    if iou_mode == 'iof':
        iofs = bbox_overlaps(pred, gt_bbox, mode='iof', is_aligned=False).clamp(min=eps)  # iof
        for i in range(iofs.shape[0]):
            interbox_idx = np.where((iofs[i, :].detach().cpu().numpy() > 0))
            interbox = []
            ious = []
            for j in range(len(interbox_idx[0])):
                interbox.append(torch.Tensor((max(pred[i, 0], gt_bbox[interbox_idx[0][j], 0]),
                                              max(pred[i, 1], gt_bbox[interbox_idx[0][j], 1]),
                                              min(pred[i, 2], gt_bbox[interbox_idx[0][j], 2]),
                                              min(pred[i, 3], gt_bbox[interbox_idx[0][j], 3]))).unsqueeze(0).cuda())
            interbox = torch.cat(interbox, dim=0)

            pred_area = (pred[i, :] * 10.0 ** 2).round()
            if (pred_area[2].int() - pred_area[0].int()) <= 0 or (pred_area[3].int() - pred_area[1].int()) <= 0:
                ious.append(torch.Tensor([0.0]).cuda())
            else:
                intersum_area = torch.zeros(
                    (pred_area[2].int() - pred_area[0].int(), pred_area[3].int() - pred_area[1].int())).cuda()
                interbox_area = (interbox * 10.0 ** 2).round()
                interbox_area[:, 0] = interbox_area[:, 0] - pred_area[0]
                interbox_area[:, 1] = interbox_area[:, 1] - pred_area[1]
                interbox_area[:, 2] = interbox_area[:, 2] - pred_area[0]
                interbox_area[:, 3] = interbox_area[:, 3] - pred_area[1]
                for j in range(interbox_area.shape[0]):
                    intersum_area[interbox_area[j, 0].int():(interbox_area[j, 2] + 1).int(),
                    interbox_area[j, 1].int():(interbox_area[j, 3] + 1).int()] = 1
                iofs_sum = torch.sum(intersum_area) / (intersum_area.shape[0] * intersum_area.shape[1])
                ious.append(iofs_sum)
        if len(ious) > 1:
            ious = torch.cat(ious)
        else:
            ious = ious[0]
    elif iou_mode == 'iog':
        iogs = bbox_overlaps(pred, gt_bbox, mode='iog', is_aligned=False).clamp(min=eps)  # iof
        for i in range(iogs.shape[1]):
            interbox_idx = np.where((iogs[:, i].detach().cpu().numpy() > 0))
            interbox = []
            ious = []
            for j in range(len(interbox_idx[0])):
                interbox.append(torch.Tensor((max(gt_bbox[i, 0], pred[interbox_idx[0][j], 0]),
                                              max(gt_bbox[i, 1], pred[interbox_idx[0][j], 1]),
                                              min(gt_bbox[i, 2], pred[interbox_idx[0][j], 2]),
                                              min(gt_bbox[i, 3], pred[interbox_idx[0][j], 3]))).unsqueeze(0).cuda())
            interbox = torch.cat(interbox, dim=0)

            gt_area = (gt_bbox[i, :] * 10.0 ** 2).round()
            intersum_area = torch.zeros(
                (gt_area[2].int() - gt_area[0].int(), gt_area[3].int() - gt_area[1].int())).cuda()
            interbox_area = (interbox * 10.0 ** 2).round()
            interbox_area[:, 0] = interbox_area[:, 0] - gt_area[0]
            interbox_area[:, 1] = interbox_area[:, 1] - gt_area[1]
            interbox_area[:, 2] = interbox_area[:, 2] - gt_area[0]
            interbox_area[:, 3] = interbox_area[:, 3] - gt_area[1]
            for j in range(interbox_area.shape[0]):
                intersum_area[interbox_area[j, 0].int():(interbox_area[j, 2] + 1).int(),
                interbox_area[j, 1].int():(interbox_area[j, 3] + 1).int()] = 1
            iogs_sum = torch.sum(intersum_area) / (intersum_area.shape[0] * intersum_area.shape[1])
            ious.append(iogs_sum)
        if len(ious) > 1:
            ious = torch.cat(ious)
        else:
            ious = ious[0]
    else:
        ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)

    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
    """BIoULoss.

    This is an implementation of paper
    `Improving Object Localization with Fitness NMS and Bounded IoU Loss.
    <https://arxiv.org/abs/1711.00164>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0]
    pred_h = pred[:, 3] - pred[:, 1]
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0]
        target_h = target[:, 3] - target[:, 1]

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) /
        (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) /
        (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w /
                            (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h /
                            (target_h + eps))
    # view(..., -1) does not work for empty tensor
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            dim=-1).flatten(1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def focal_eiou_loss(box1, box2, gamma=0.5, alpha=1, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2

    rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
    rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
    cw2 = torch.pow(cw ** 2 + eps, alpha)
    ch2 = torch.pow(ch ** 2 + eps, alpha)
    iou_value = iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(iou, gamma) # Focal_EIou
    loss_iou = (1.0 - iou_value[0]) * iou_value[1].detach()
    return loss_iou

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def piou_loss(box1, box2, Lambda=1.3, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height

    # PIoU
    dw1 = torch.abs(b1_x2.minimum(b1_x1) - b2_x2.minimum(b2_x1))
    dw2 = torch.abs(b1_x2.maximum(b1_x1) - b2_x2.maximum(b2_x1))
    dh1 = torch.abs(b1_y2.minimum(b1_y1) - b2_y2.minimum(b2_y1))
    dh2 = torch.abs(b1_y2.maximum(b1_y1) - b2_y2.maximum(b2_y1))
    P = ((dw1 + dw2) / torch.abs(w2) + (dh1 + dh2) / torch.abs(h2)) / 4
    L_v1 = 1 - iou - torch.exp(-P ** 2) + 1

    PIoU = False
    PIoU2 = True
    if PIoU:
        return L_v1

    if PIoU2:
        q = torch.exp(-P)
        x = q * Lambda
        return 3 * x * torch.exp(-x ** 2) * L_v1

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def shape_iou(box1, box2, scale=0.8, eps=1e-7):
    (b1_x1, b1_x2, b1_y1, b1_y2), (b2_x1, b2_x2, b2_y1, b2_y2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    # Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance
    ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
    hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
    center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    center_distance = hh * center_distance_x + ww * center_distance_y
    distance = center_distance / c2

    # Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape
    omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)

    # Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU
    iou = iou - distance - 0.5 * (shape_cost)
    return 1 - iou  # IoU

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def wiou_loss(pred_boxes, target_boxes, weight=None, eps=1e-7):
    """
    Compute the Weighted IoU loss between predicted and target bounding boxes.

    其中，输入pred_boxes和target_boxes分别是形状为(N, 4)的预测边界框和目标边界框张量。
    如果需要使用权重，则输入形状为(N,)的权重张量weight，否则默认为None。函数返回一个标量，表示计算出的加权IoU损失。
    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes, with shape (N, 4).
        target_boxes (torch.Tensor): Target bounding boxes, with shape (N, 4).
        weight (torch.Tensor, optional): Weight tensor with shape (N,). Defaults to None.

    Returns:
        torch.Tensor: Weighted IoU loss scalar.
    """
    # Compute the intersection over union (IoU) between the predicted and target boxes.
    x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    pred_boxes_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_boxes_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_boxes_area + target_boxes_area - intersection_area
    iou = intersection_area / (union_area + eps)

    # Compute the Weighted IoU loss using the IoU and weight tensors.
    if weight is None:
        w_iou = 1 - iou.mean()
    else:
        w_iou = (1 - iou) * weight
        w_iou = w_iou.sum() / weight.sum()

    return w_iou

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def siou_loss(pred, target, eps=1e-7, neg_gamma=True):
    r"""`Implementation of paper `SIoU Loss: More Powerful Learning
    for Bounding Box Regression <https://arxiv.org/abs/2205.12740>`_.

    Code is modified from https://github.com/meituan/YOLOv6.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
        neg_gamma (bool): `True` follows original implementation in paper.

    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    # modified clamp threshold zero to eps to avoid NaN
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=eps)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # angle cost
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps

    sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)

    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = torch.cos(torch.asin(sin_alpha) * 2 - math.pi / 2)

    # distance cost
    rho_x = (s_cw / cw)**2
    rho_y = (s_ch / ch)**2

    # `neg_gamma=True` follows original implementation in paper
    # but setting `neg_gamma=False` makes training more stable.
    gamma = angle_cost - 2 if neg_gamma else 2 - angle_cost
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

    # shape cost
    omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(
        1 - torch.exp(-1 * omiga_h), 4)

    # SIoU
    sious = ious - 0.5 * (distance_cost + shape_cost)
    loss = 1 - sious.clamp(min=-1.0, max=1.0)
    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def inner_ciou(box1, box2, ratio = 1.0, eps=1e-7):
    (b1_x1, b1_x2, b1_y1, b1_y2), (b2_x1, b2_x2, b2_y1, b2_y2) = box1.chunk(4, -1), box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    x1, x2, y1, y2 = b1_x1 + w1_, b2_x1 + w2_, b1_y1 + h1_, b2_y1 + h2_
    #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU       #IoU      #IoU
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
   #Inner-IoU      #Inner-IoU        #Inner-IoU        #Inner-IoU        #Inner-IoU        #Inner-IoU        #Inner-IoU
    inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_*ratio, x1 + w1_*ratio,\
                                                             y1 - h1_*ratio, y1 + h1_*ratio
    inner_b2_x1,inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_*ratio, x2 + w2_*ratio,\
                                                             y2 - h2_*ratio, y2 + h2_*ratio
    inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
                   (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
    inner_union = w1*ratio * h1*ratio + w2*ratio * h2*ratio - inner_inter + eps
    inner_iou = inner_inter/inner_union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex  width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return 1 - inner_iou + (rho2 / c2 + v * alpha)

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def eiou_loss(pred,
              target,
              smooth_point = 0.1,
              eps = 1e-7):
    r"""Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        smooth_point (float): hyperparameter, default is 0.1.
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # extent top left
    ex1 = torch.min(px1, tx1)
    ey1 = torch.min(py1, ty1)

    # intersection coordinates
    ix1 = torch.max(px1, tx1)
    iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2)
    iy2 = torch.min(py2, ty2)

    # extra
    xmin = torch.min(ix1, ix2)
    ymin = torch.min(iy1, iy2)
    xmax = torch.max(ix1, ix2)
    ymax = torch.max(iy1, iy2)

    # Intersection
    intersection = (ix2 - ex1) * (iy2 - ey1) + (xmin - ex1) * (ymin - ey1) - (
        ix1 - ex1) * (ymax - ey1) - (xmax - ex1) * (
            iy1 - ey1)
    # Union
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (
        ty2 - ty1) - intersection + eps
    # IoU
    ious = 1 - (intersection / union)

    # Smooth-EIoU
    smooth_sign = (ious < smooth_point).detach().float()
    loss = 0.5 * smooth_sign * (ious**2) / smooth_point + (1 - smooth_sign) * (
        ious - 0.5 * smooth_point)
    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def adiou_loss(box1, box2, a=3, eps=1e-7):

    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, 1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, 1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    diou = rho2 / c2
    adiou = torch.pow(diou, a)
    return 1 - iou + adiou

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def diou_loss(pred, target, eps=1e-7):
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.

    Code is modified from https://github.com/Zzh-tju/DIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union
    #alpha iou
##    alpha = 3
##    ious = torch.pow(overlap / union, alpha)

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    # DIoU
    dious = ious - rho2 / c2
    loss = 1 - dious
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def ciou_loss(pred, target, eps=1e-7):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps
##    union = ag + eps    #ciof

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    with torch.no_grad():
        alpha = (ious > 0.5).float() * v / (1 - ious + v)

    # CIoU
    cious = ious - (rho2 / c2 + alpha * v)
    loss = 1 - cious.clamp(min=-1.0, max=1.0)
    return loss


def area_prov3_9_4(alpha):
    '''
    if alpha <= 1:
        alpha_pro = (alpha - 1).pow(2)
    elif (alpha > 1) and (alpha <= 1.5):
        alpha_pro = (math.pi / 8 + 0.0625) * 0.5 * (alpha - 1).pow(2)
    else:
        alpha_pro = (math.pi / 4 + 0.125) * torch.atan(alpha - 0.5) - math.pi / 4 + 0.125
    '''
    alpha_pro1 = (alpha <= 1).float()
    alpha_pro2 = ((alpha > 1) & (alpha <= 2.25)).float()
    alpha_pro3 = (alpha > 2.25).float()
    alpha_area1 = (alpha - 1).pow(2) * alpha_pro1
    alpha_area2 = 16 / (25 + 20 * math.pi) * (alpha - 1).pow(2) * alpha_pro2
    alpha_area3 = (16 / (5 + 4 * math.pi) * torch.atan(alpha - 1.25) + (5 - 4 * math.pi) / (5 + 4 * math.pi)) * alpha_pro3
    alpha_area = alpha_area1 + alpha_area2 + alpha_area3
    return alpha_area



@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def casiou_loss(pred, target, eps=1e-7, mode='iou', gt_bbox=None):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    if mode == 'iou':
        union = ap + ag - overlap + eps
    elif mode == 'iog':
        union = ag + eps    #ciof
    else:
        union = ap + eps  # ciof

    if mode == 'iof':
        iofs = bbox_overlaps(pred, gt_bbox, mode='iof', is_aligned=False).clamp(min=eps)  # iof
##        iofs_sum = torch.sum(iofs, dim=1).clamp(max=1.0)
        for i in range(iofs.shape[0]):
            interbox_idx = np.where((iofs[i, :].detach().cpu().numpy() > 0))
            interbox = []
            ious = []
            for j in range(len(interbox_idx[0])):
                interbox.append(torch.Tensor((max(pred[i, 0], gt_bbox[interbox_idx[0][j], 0]),
                                 max(pred[i, 1], gt_bbox[interbox_idx[0][j], 1]),
                                 min(pred[i, 2], gt_bbox[interbox_idx[0][j], 2]),
                                 min(pred[i, 3], gt_bbox[interbox_idx[0][j], 3]))).unsqueeze(0).cuda())
            interbox = torch.cat(interbox, dim=0)

            pred_area = (pred[i,:] * 10.0**2).round()
            if (pred_area[2].int() - pred_area[0].int()) <= 0 or (pred_area[3].int() - pred_area[1].int()) <= 0:
                ious.append(torch.Tensor([0.0]).cuda())
            else:
                intersum_area = torch.zeros((pred_area[2].int() - pred_area[0].int(), pred_area[3].int() - pred_area[1].int())).cuda()
                interbox_area = (interbox * 10.0**2).round()
                interbox_area[:, 0] = interbox_area[:, 0] - pred_area[0]
                interbox_area[:, 1] = interbox_area[:, 1] - pred_area[1]
                interbox_area[:, 2] = interbox_area[:, 2] - pred_area[0]
                interbox_area[:, 3] = interbox_area[:, 3] - pred_area[1]
                for j in range(interbox_area.shape[0]):
                    intersum_area[interbox_area[j, 0].int():(interbox_area[j, 2]+1).int(), interbox_area[j, 1].int():(interbox_area[j, 3]+1).int()] = 1
                iofs_sum = torch.sum(intersum_area) / (intersum_area.shape[0] * intersum_area.shape[1])
                ious.append(iofs_sum)
        if len(ious) > 1:
            ious = torch.cat(ious)
        else:
            ious = ious[0]
    elif mode == 'iog':
        iogs = bbox_overlaps(pred, gt_bbox, mode='iog', is_aligned=False).clamp(min=eps)  # iof
##        iogs_sum = torch.sum(iogs, dim=0)
        for i in range(iogs.shape[1]):
            interbox_idx = np.where((iogs[:, i].detach().cpu().numpy() > 0))
            interbox = []
            ious = []
            for j in range(len(interbox_idx[0])):
                interbox.append(torch.Tensor((max(gt_bbox[i, 0], pred[interbox_idx[0][j], 0]),
                                 max(gt_bbox[i, 1], pred[interbox_idx[0][j], 1]),
                                 min(gt_bbox[i, 2], pred[interbox_idx[0][j], 2]),
                                 min(gt_bbox[i, 3], pred[interbox_idx[0][j], 3]))).unsqueeze(0).cuda())
            interbox = torch.cat(interbox, dim=0)

            gt_area = (gt_bbox[i,:] * 10.0**2).round()
            intersum_area = torch.zeros((gt_area[2].int() - gt_area[0].int(), gt_area[3].int() - gt_area[1].int())).cuda()
            interbox_area = (interbox * 10.0**2).round()
            interbox_area[:, 0] = interbox_area[:, 0] - gt_area[0]
            interbox_area[:, 1] = interbox_area[:, 1] - gt_area[1]
            interbox_area[:, 2] = interbox_area[:, 2] - gt_area[0]
            interbox_area[:, 3] = interbox_area[:, 3] - gt_area[1]
            for j in range(interbox_area.shape[0]):
                intersum_area[interbox_area[j, 0].int():(interbox_area[j, 2]+1).int(), interbox_area[j, 1].int():(interbox_area[j, 3]+1).int()] = 1
            iogs_sum = torch.sum(intersum_area) / (intersum_area.shape[0] * intersum_area.shape[1])
            ious.append(iogs_sum)

        if len(ious) > 1:
            ious = torch.cat(ious)
        else:
            ious = ious[0]

    ap = torch.clamp(ap, min=1e-8)
    area_bi = torch.div(ag, ap)

    # IoU
    if mode in ['iou', 'giou']:
        ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw**2 + ch**2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
    rho2 = left + right

    factor = 4 / math.pi**2
    v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

    with torch.no_grad():
        alpha_ciou = (ious > 0.5).float() * v / (1 - ious + v + eps)

        area_prop = area_prov3_9_4(area_bi)
        alpha_pro = area_prop / (1 - ious + area_prop + eps)

    # CIoU
    cious = ious - (rho2 / c2 + alpha_ciou * v + alpha_pro * area_prop)
    loss = 1 - cious.clamp(min=-1.0, max=1.0)
    return loss


@LOSSES.register_module()
class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='log',
                 iou_mode='iof'):
        super(IoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.iou_mode = iou_mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            iou_mode=self.iou_mode,
            gt_bbox=gt_bbox,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class BoundedIoULoss(nn.Module):

    def __init__(self, beta=0.2, eps=1e-3, reduction='mean', loss_weight=1.0):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * bounded_iou_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class GIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class DIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(DIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * diou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class CIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(CIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * ciou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@LOSSES.register_module()
class CASIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0, mode='iou'):
        super(CASIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.mode = mode

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * casiou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            mode=self.mode,
            gt_bbox=gt_bbox,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class WIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(WIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * wiou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@LOSSES.register_module()
class EIoULoss(nn.Module):
    r"""Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        smooth_point (float): hyperparameter, default is 0.1.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 smooth_point: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smooth_point = smooth_point

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight = None,
                avg_factor = None,
                reduction_override = None,
                **kwargs):
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * eiou_loss(
            pred,
            target,
            weight,
            smooth_point=self.smooth_point,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class FocalEIoULoss(nn.Module):
    r"""Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        smooth_point (float): hyperparameter, default is 0.1.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 smooth_point: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smooth_point = smooth_point

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight = None,
                avg_factor = None,
                reduction_override = None,
                **kwargs):
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * focal_eiou_loss(
            pred,
            target,
##            smooth_point=self.smooth_point,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@LOSSES.register_module()
class SIoULoss(nn.Module):
    r"""`Implementation of paper `SIoU Loss: More Powerful Learning
    for Bounding Box Regression <https://arxiv.org/abs/2205.12740>`_.

    Code is modified from https://github.com/meituan/YOLOv6.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
        neg_gamma (bool): `True` follows original implementation in paper.

    Return:
        Tensor: Loss tensor.
    """

    def __init__(self,
                 eps = 1e-6,
                 reduction = 'mean',
                 loss_weight = 1.0,
                 neg_gamma = False):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.neg_gamma = neg_gamma

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight = None,
                avg_factor = None,
                reduction_override = None,
                **kwargs):
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * siou_loss(
            pred,
            target,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            neg_gamma=self.neg_gamma,
            **kwargs)
        return loss

@LOSSES.register_module()
class PIoULoss(nn.Module):
    r"""Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        smooth_point (float): hyperparameter, default is 0.1.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 smooth_point: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smooth_point = smooth_point

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight = None,
                avg_factor = None,
                reduction_override = None,
                **kwargs):
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * piou_loss(
            pred,
            target,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@LOSSES.register_module()
class ShapeIoULoss(nn.Module):
    r"""Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        smooth_point (float): hyperparameter, default is 0.1.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 smooth_point: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smooth_point = smooth_point

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight = None,
                avg_factor = None,
                reduction_override = None,
                **kwargs):
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * shape_iou(
            pred,
            target,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@LOSSES.register_module()
class ADIoULoss(nn.Module):
    r"""Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        smooth_point (float): hyperparameter, default is 0.1.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 smooth_point: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smooth_point = smooth_point

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight = None,
                avg_factor = None,
                reduction_override = None,
                **kwargs):
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * adiou_loss(
            pred,
            target,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss

@LOSSES.register_module()
class InnerCIoULoss(nn.Module):
    r"""Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        smooth_point (float): hyperparameter, default is 0.1.
    """

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 smooth_point: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.smooth_point = smooth_point

    def forward(self,
                pred,
                target,
                gt_bbox,
                weight = None,
                avg_factor = None,
                reduction_override = None,
                **kwargs):
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * inner_ciou(
            pred,
            target,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss