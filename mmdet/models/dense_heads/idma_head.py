# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.ops import deform_conv2d
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, distance2bbox,
                        images_to_levels, multi_apply, reduce_mean, unmap, bbox_overlaps)
from mmdet.core.utils import filter_scores_and_topk
from mmdet.models.utils import sigmoid_geometric_mean
from ..builder import HEADS, build_loss
from .atss_head import ATSSHead

from mmcv.ops import DeformConv2d
import numpy as np
import math
import copy

from ..necks.ssfpn_changed import DAN_PAM, DAN_CAM
from timm.models.layers import trunc_normal_
from matplotlib import pyplot as plt

from mmcv.cnn import NonLocal2d

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PicanetL(nn.Module):
    def __init__(self, in_channel):
        super(PicanetL, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 128, kernel_size=7, dilation=2, padding=6)
        self.conv2 = nn.Conv2d(128, 49, kernel_size=1)

    def forward(self, *input):
        x = input[0]
        size = x.size()
        kernel = self.conv1(x)
        kernel = self.conv2(kernel)
        kernel = F.softmax(kernel, 1)
        kernel = kernel.reshape(size[0], 1, size[2] * size[3], 7 * 7)
        # print("Before unfold", x.shape)
        x = F.unfold(x, kernel_size=[7, 7], dilation=[2, 2], padding=6)
        # print("After unfold", x.shape)
        x = x.reshape(size[0], size[1], size[2] * size[3], -1)
        # print(x.shape, kernel.shape)
        x = torch.mul(x, kernel)
        x = torch.sum(x, dim=3)
        x = x.reshape(size[0], size[1], size[2], size[3])
        return x

class acf_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_channels, out_channels):
        super(acf_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True),
                                nn.Dropout2d(0.2, False))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, feat_ffm, coarse_x):
        """
            inputs :
                feat_ffm : input feature maps( B X C X H X W), C is channel
                coarse_x : input feature maps( B X N X H X W), N is class
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, N, height, width = coarse_x.size()

        # CCB: Class Center Block start...
        # 1x1conv -> F'
        feat_ffm = self.conv1(feat_ffm)
        b, C, h, w = feat_ffm.size()

        # P_coarse reshape ->(B, N, W*H)
        proj_query = coarse_x.view(m_batchsize, N, -1)

        # F' reshape and transpose -> (B, W*H, C')
        proj_key = feat_ffm.view(b, C, -1).permute(0, 2, 1)

        # multiply & normalize ->(B, N, C')
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        # CCB: Class Center Block end...

        # CAB: Class Attention Block start...
        # transpose ->(B, C', N)
        attention = attention.permute(0, 2, 1)

        # (B, N, W*H)
        proj_value = coarse_x.view(m_batchsize, N, -1)

        # # multiply (B, C', N)(B, N, W*H)-->(B, C, W*H)
        out = torch.bmm(attention, proj_value)

        out = out.view(m_batchsize, C, height, width)

        # 1x1conv
        out = self.conv2(out)
        # CAB: Class Attention Block end...

        return out

class ACFModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ACFModule, self).__init__()

        self.acf = acf_Module(in_channels, out_channels)


    def forward(self, x, coarse_x):
        class_output = self.acf(x, coarse_x)
        # feat_cat = torch.cat([class_output, output],dim=1)
        return class_output

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
#        self.bn1 = nn.BatchNorm2d(in_planes // 16)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#        self.bn2 = nn.BatchNorm2d(in_planes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))    #
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))    #
        out = avg_out + max_out
        return self.sigmoid(out)    #self.sigmoid(w_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):    #, reduction=16, drop_rate=0.3
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self,
                 feat_channels,
                 stacked_convs,
                 la_down_rate=8,
                 conv_cfg=None,
                 norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.stacked_convs,
                1,
                padding=0), nn.Sigmoid())

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=norm_cfg is None)

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)

        return feat


@HEADS.register_module()
class IDMAHead(ATSSHead):
    """

    Args:
        num_dcn (int): Number of deformable convolution in the head.
            Default: 0.
        anchor_type (str): If set to `anchor_free`, the head will use centers
            to regress bboxes. If set to `anchor_based`, the head will
            regress bboxes based on anchors. Default: `anchor_free`.
        initial_loss_cls (dict): Config of initial loss.

    Example:
        >>> self = TOODHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_dcn=0,
                 anchor_type='anchor_free',
                 initial_loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     activated=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_last=dict(type='CASIoULoss', loss_weight=2.0, mode='iou'),  #fine tunning loss iou
                 loss_iof=dict(type='CASIoULoss', loss_weight=2.0, mode='iof',),  #loss iof
                 loss_iog=dict(type='IoULoss', loss_weight=2.0, mode='linear', iou_mode='iog'),
                 **kwargs):
        assert anchor_type in ['anchor_free', 'anchor_based']

        self.num_dcn = num_dcn
        self.anchor_type = anchor_type
        self.epoch = 0  # which would be update in SetEpochInfoHook!
        super(IDMAHead, self).__init__(num_classes, in_channels, **kwargs)

        if self.train_cfg:
            self.initial_epoch = self.train_cfg.initial_epoch
            self.initial_assigner = build_assigner(
                self.train_cfg.initial_assigner)
            self.initial_loss_cls = build_loss(initial_loss_cls)
            self.assigner = self.initial_assigner
            self.alignment_assigner = build_assigner(self.train_cfg.assigner)
            self.alpha = self.train_cfg.alpha
            self.beta = self.train_cfg.beta

            self.loss_bbox_last = build_loss(loss_bbox_last)    #loss iou
            self.iof_iog = False
            self.loss_iof = build_loss(loss_iof)    #loss iof
            self.loss_iog = build_loss(loss_iog)    #loss iog

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.inter_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            if i < self.num_dcn:
                conv_cfg = dict(type='DCNv2', deform_groups=4)
            else:
                conv_cfg = self.conv_cfg
            chn = self.in_channels if i == 0 else self.feat_channels

            self.inter_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.cls_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            self.conv_cfg, self.norm_cfg)
        self.reg_decomp = TaskDecomposition(self.feat_channels,
                                            self.stacked_convs,
                                            self.stacked_convs * 8,
                                            self.conv_cfg, self.norm_cfg)

        self.tood_cls = nn.Conv2d(
            self.feat_channels,
            self.num_base_priors * self.cls_out_channels,
            3,
            padding=1)
        self.tood_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4 * self.cls_out_channels, 3, padding = 1)  # per-class

        self.cls_prob_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs,
                      self.feat_channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 1, 3, padding=1))
        self.reg_offset_module = nn.Sequential(
            nn.Conv2d(self.feat_channels * self.stacked_convs,
                      self.feat_channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_channels // 4, 4 * 2 * self.cls_out_channels, 3, padding=1))  #per-class
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.prior_generator.strides])

        #up fusion
        self.cls_prob_cat_up = nn.ModuleList(
            [nn.Conv2d(2, 1, 1)
            for _ in range(len(self.prior_generator.strides) - 1)])
        for i in range(self.cls_out_channels):
            logits_cat_up = nn.ModuleList(
                            [nn.Conv2d(2, 1, 1)
                            for _ in range(len(self.prior_generator.strides) - 1)])
            setattr(self, f"logits_cat_up{i}", logits_cat_up)


        #down fusion
        self.cls_prob_cat_down = nn.ModuleList(
            [nn.Conv2d(2, 1, 1)
            for _ in range(len(self.prior_generator.strides) - 1)])
        for i in range(self.cls_out_channels):
            logits_cat_down = nn.ModuleList(
                            [nn.Conv2d(2, 1, 1)
                            for _ in range(len(self.prior_generator.strides) - 1)])
            setattr(self, f"logits_cat_down{i}", logits_cat_down)


        #up+down
        self.cls_prob_cat = nn.ModuleList(
            [nn.Conv2d(2, 1, 1)
            for _ in range(len(self.prior_generator.strides))])
        for i in range(self.cls_out_channels):
            logits_cat = nn.ModuleList(
                         [nn.Conv2d(2, 1, 1)
                         for _ in range(len(self.prior_generator.strides))])
            setattr(self, f"logits_cat{i}", logits_cat)

        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)

        for m in self.inter_convs:
            normal_init(m.conv, std=0.01)

        for m in self.cls_prob_module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_offset_module:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.cls_prob_module[-1], std=0.01, bias=bias_cls)

        self.cls_decomp.init_weights()
        self.reg_decomp.init_weights()

        normal_init(self.tood_cls, std=0.01, bias=bias_cls)
        normal_init(self.tood_reg, std=0.01)

        #up fusion
        normal_init(self.cls_prob_cat_up, std=0.01, bias=bias_cls)
        for i in range(self.cls_out_channels):
            logits_cat_up = getattr(self, f"logits_cat_up{i}")
            normal_init(logits_cat_up, std=0.01, bias=bias_cls)


        #down fusion
        normal_init(self.cls_prob_cat_down, std=0.01, bias=bias_cls)
        for i in range(self.cls_out_channels):
            logits_cat_down = getattr(self, f"logits_cat_down{i}")
            normal_init(logits_cat_down, std=0.01, bias=bias_cls)


        #up+down
        normal_init(self.cls_prob_cat, std=0.01, bias=bias_cls)
        for i in range(self.cls_out_channels):
            logits_cat = getattr(self, f"logits_cat{i}")
            normal_init(logits_cat, std=0.01, bias=bias_cls)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification scores for all scale
                    levels, each is a 4D-tensor, the channels number is
                    num_anchors * num_classes.
                bbox_preds (list[Tensor]): Decoded box for all scale levels,
                    each is a 4D-tensor, the channels number is
                    num_anchors * 4. In [tl_x, tl_y, br_x, br_y] format.
        """
        cls_scores = []
        bbox_preds = []
        bbox_preds_refine = []

        anchor_list, feat_list, reg_feat_list, cls_logits_list, cls_prob_list = [], [], [], [], []

        for idx, (x, scale, stride) in enumerate(zip(feats, self.scales, self.prior_generator.strides)):

            b, c, h, w = x.shape
            anchor = self.prior_generator.single_level_grid_priors(
                (h, w), idx, device=x.device)
            anchor = torch.cat([anchor for _ in range(b)])
            # extract task interactive features
            inter_feats = []
            for inter_conv in self.inter_convs:
                x = inter_conv(x)
                inter_feats.append(x)
            feat = torch.cat(inter_feats, 1)

            # task decomposition
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_feat = self.cls_decomp(feat, avg_feat)

            reg_feat = self.reg_decomp(feat, avg_feat)

            # cls prediction and alignment
            cls_logits = self.tood_cls(cls_feat)
            cls_prob = self.cls_prob_module(feat)

            anchor_list.append(anchor)
            feat_list.append(feat)
            reg_feat_list.append(reg_feat)

            cls_logits_list.append(cls_logits)

            cls_prob_list.append(cls_prob)

        fusion_mode = 'step'    #'normal'   'step'
        attention_mode = False
        add_mode = 'cat' #'cat' 'add'

        # down fusion
        cls_logits_cat_down_list, cls_prob_cat_down_list = [], []
        for i in range(len(feat_list) - 1):
            down_shape = cls_prob_list[i + 1].shape[2:]
            if fusion_mode == 'normal':
                if add_mode == 'cat':
                    cls_prob_cat_down = torch.cat((F.interpolate(cls_prob_list[i], size=down_shape, mode='nearest'),
                                                   cls_prob_list[i + 1]), dim=1)
                elif add_mode == 'add':
                    cls_prob_cat_down = F.interpolate(cls_prob_list[i], size=down_shape, mode='nearest') + \
                                        cls_prob_list[i + 1]
            elif fusion_mode == 'step':
                # up+down step
                if i == 0:
                    if add_mode == 'cat':
                        cls_prob_cat_down = torch.cat((F.interpolate(cls_prob_list[i], size=down_shape, mode='nearest'),
                                                       cls_prob_list[i + 1]), dim=1)
                    elif add_mode == 'add':
                        cls_prob_cat_down = F.interpolate(cls_prob_list[i], size=down_shape, mode='nearest') + \
                                            cls_prob_list[i + 1]
                else:
                    if add_mode == 'cat':
                        cls_prob_cat_down = torch.cat(
                            (F.interpolate(cls_prob_cat_down_list[i - 1], size=down_shape, mode='nearest'),
                             cls_prob_list[i + 1]), dim=1)
                    elif add_mode == 'add':
                        cls_prob_cat_down = F.interpolate(cls_prob_cat_down_list[i - 1], size=down_shape,
                                                          mode='nearest') + cls_prob_list[i + 1]
            if add_mode == 'cat':
                prob_conv_feat_down = self.cls_prob_cat_down[i](cls_prob_cat_down)
            elif add_mode == 'add':
                prob_conv_feat_down = cls_prob_cat_down
            if attention_mode:
                # fusion attention
                prob_conv_feat_down = (1 + self.sigmoid(prob_conv_feat_down)) * cls_prob_list[i + 1]

            cls_prob_cat_down_list.append(prob_conv_feat_down)

            logits_conv_feat_cls_down_list = []
            for j in range(cls_logits.shape[1]):
                if add_mode == 'cat':
                    logits_cat_down = getattr(self, f"logits_cat_down{j}")[i]
                if fusion_mode == 'normal':
                    if add_mode == 'cat':
                        cls_logits_cat_down = torch.cat(
                            (
                                F.interpolate(cls_logits_list[i][:, j, :, :].unsqueeze(1), size=down_shape,
                                              mode='nearest'),
                                cls_logits_list[i + 1][:, j, :, :].unsqueeze(1)), dim=1)
                    elif add_mode == 'add':
                        cls_logits_cat_down = F.interpolate(cls_logits_list[i][:, j, :, :].unsqueeze(1),
                                                            size=down_shape, mode='nearest') + cls_logits_list[i + 1][:,
                                                                                               j, :, :].unsqueeze(1)
                elif fusion_mode == 'step':
                    # up+down step
                    if i == 0:
                        if add_mode == 'cat':
                            cls_logits_cat_down = torch.cat(
                                (F.interpolate(cls_logits_list[i][:, j, :, :].unsqueeze(1), size=down_shape,
                                               mode='nearest'),
                                 cls_logits_list[i + 1][:, j, :, :].unsqueeze(1)), dim=1)
                        elif add_mode == 'add':
                            cls_logits_cat_down = F.interpolate(cls_logits_list[i][:, j, :, :].unsqueeze(1),
                                                                size=down_shape, mode='nearest') + cls_logits_list[
                                                                                                       i + 1][:, j, :,
                                                                                                   :].unsqueeze(1)
                    else:
                        if add_mode == 'cat':
                            cls_logits_cat_down = torch.cat(
                                (
                                    F.interpolate(cls_logits_cat_down_list[i - 1][:, j, :, :].unsqueeze(1),
                                                  size=down_shape,
                                                  mode='nearest'),
                                    cls_logits_list[i + 1][:, j, :, :].unsqueeze(1)), dim=1)
                        elif add_mode == 'add':
                            cls_logits_cat_down = F.interpolate(
                                cls_logits_cat_down_list[i - 1][:, j, :, :].unsqueeze(1), size=down_shape,
                                mode='nearest') + cls_logits_list[i + 1][:, j, :, :].unsqueeze(1)
                if add_mode == 'cat':
                    logits_conv_feat_cls_down = logits_cat_down(cls_logits_cat_down)
                elif add_mode == 'add':
                    logits_conv_feat_cls_down = cls_logits_cat_down
                if attention_mode:
                    # fusion attention
                    logits_conv_feat_cls_down = (1 + self.sigmoid(logits_conv_feat_cls_down)) * cls_logits_list[i + 1][
                                                                                                :, j, :, :].unsqueeze(1)

                logits_conv_feat_cls_down_list.append(logits_conv_feat_cls_down)
            logits_conv_feat_down = torch.cat(logits_conv_feat_cls_down_list, dim=1)

            cls_logits_cat_down_list.append(logits_conv_feat_down)

        #up fusion
        cls_logits_cat_up_list, cls_prob_cat_up_list = [], []
        for i in reversed(range(len(feat_list))[1:]):
            up_shape = cls_prob_list[i - 1].shape[2:]
            if fusion_mode == 'normal':
                if add_mode == 'cat':
                    cls_prob_cat_up = torch.cat((F.interpolate(cls_prob_list[i], size=up_shape, mode='nearest'),
                                                 cls_prob_list[i - 1]), dim=1)
                elif add_mode == 'add':
                    cls_prob_cat_up = F.interpolate(cls_prob_list[i], size=up_shape, mode='nearest') + cls_prob_list[
                        i - 1]

            elif fusion_mode == 'step':
                # up+down step
                if i == (len(feat_list) - 1):
                    if add_mode == 'cat':
                        cls_prob_cat_up = torch.cat((F.interpolate(cls_prob_list[i], size=up_shape, mode='nearest'),
                                                     cls_prob_list[i - 1]), dim=1)
                    elif add_mode == 'add':
                        cls_prob_cat_up = F.interpolate(cls_prob_list[i], size=up_shape, mode='nearest') + \
                                          cls_prob_list[i - 1]
                else:
                    if add_mode == 'cat':
                        cls_prob_cat_up = torch.cat(
                            (F.interpolate(cls_prob_cat_up_list[len(feat_list) - 2 - i], size=up_shape, mode='nearest'),
                             cls_prob_list[i - 1]), dim=1)
                    elif add_mode == 'add':
                        cls_prob_cat_up = F.interpolate(cls_prob_cat_up_list[len(feat_list) - 2 - i], size=up_shape,
                                                        mode='nearest') + cls_prob_list[i - 1]

            if add_mode == 'add':
                prob_conv_feat_up = cls_prob_cat_up
            elif add_mode == 'cat':
                prob_conv_feat_up = self.cls_prob_cat_up[i - 1](cls_prob_cat_up)
            if attention_mode:
                # fusion attention
                prob_conv_feat_up = (1 + self.sigmoid(prob_conv_feat_up)) * cls_prob_list[i - 1]

            cls_prob_cat_up_list.append(prob_conv_feat_up)

            logits_conv_feat_cls_up_list = []
            for j in range(cls_logits.shape[1]):
                if add_mode == 'cat':
                    logits_cat_up = getattr(self, f"logits_cat_up{j}")[i - 1]
                if fusion_mode == 'normal':
                    if add_mode == 'cat':
                        cls_logits_cat_up = torch.cat(
                            (F.interpolate(cls_logits_list[i][:, j, :, :].unsqueeze(1), size=up_shape, mode='nearest'),
                             cls_logits_list[i - 1][:, j, :, :].unsqueeze(1)), dim=1)
                    elif add_mode == 'add':
                        cls_logits_cat_up = F.interpolate(cls_logits_list[i][:, j, :, :].unsqueeze(1), size=up_shape,
                                                          mode='nearest') + cls_logits_list[i - 1][:, j, :,
                                                                            :].unsqueeze(1)
                elif fusion_mode == 'step':
                    # up+down step
                    if i == (len(feat_list) - 1):
                        if add_mode == 'cat':
                            cls_logits_cat_up = torch.cat(
                                (F.interpolate(cls_logits_list[i][:, j, :, :].unsqueeze(1), size=up_shape,
                                               mode='nearest'),
                                 cls_logits_list[i - 1][:, j, :, :].unsqueeze(1)), dim=1)
                        elif add_mode == 'add':
                            cls_logits_cat_up = F.interpolate(cls_logits_list[i][:, j, :, :].unsqueeze(1),
                                                              size=up_shape, mode='nearest') + cls_logits_list[i - 1][:,
                                                                                               j, :, :].unsqueeze(1)
                    else:
                        if add_mode == 'cat':
                            cls_logits_cat_up = torch.cat(
                                (F.interpolate(cls_logits_cat_up_list[len(feat_list) - 2 - i][:, j, :, :].unsqueeze(1),
                                               size=up_shape, mode='nearest'),
                                 cls_logits_list[i - 1][:, j, :, :].unsqueeze(1)), dim=1)
                        elif add_mode == 'add':
                            cls_logits_cat_up = F.interpolate(
                                cls_logits_cat_up_list[len(feat_list) - 2 - i][:, j, :, :].unsqueeze(1),
                                size=up_shape, mode='nearest') + cls_logits_list[i - 1][:, j, :, :].unsqueeze(1)
                if add_mode == 'cat':
                    logits_conv_feat_cls_up = logits_cat_up(cls_logits_cat_up)
                elif add_mode == 'add':
                    logits_conv_feat_cls_up = cls_logits_cat_up
                if attention_mode:
                    # fusion attention
                    logits_conv_feat_cls_up = (1 + self.sigmoid(logits_conv_feat_cls_up)) * cls_logits_list[i - 1][:, j,
                                                                                            :, :].unsqueeze(1)

                logits_conv_feat_cls_up_list.append(logits_conv_feat_cls_up)
            logits_conv_feat_up = torch.cat(logits_conv_feat_cls_up_list, dim=1)

            cls_logits_cat_up_list.append(logits_conv_feat_up)

        #up+down
        fu_add_mode = 'cat'    #add cat
        fu_attention_mode = True
        gate_mode = False

        cls_prob_cat_up_list.insert(0, cls_prob_list[len(feat_list)-1])
        cls_logits_cat_up_list.insert(0, cls_logits_list[len(feat_list)-1])
        cls_prob_cat_down_list.insert(0, cls_prob_list[0])
        cls_logits_cat_down_list.insert(0, cls_logits_list[0])

        cls_logits_cat_list, cls_prob_cat_list = [], []
        for i in range(len(feat_list)):
            if fu_add_mode == 'cat':
                cls_prob_cat = torch.cat((cls_prob_cat_up_list[::-1][i], cls_prob_cat_down_list[i]), dim=1)
                prob_conv_feat = self.cls_prob_cat[i](cls_prob_cat)
            elif fu_add_mode == 'add':
                #add_gate
                if gate_mode == True:
                    prob_conv_feat = self.cls_prob_up_add[i](cls_prob_cat_up_list[::-1][i]) + self.cls_prob_down_add[i](cls_prob_cat_down_list[i])
                else:
                    prob_conv_feat = cls_prob_cat_up_list[::-1][i] + cls_prob_cat_down_list[i]
            #fusion attention
            if fu_attention_mode == True:
                prob_conv_feat = (1 + self.sigmoid(prob_conv_feat)) * cls_prob_list[i]

            cls_prob_cat_list.append(prob_conv_feat)

            logits_conv_feat_cls_list = []
            for j in range(cls_logits.shape[1]):
                if fu_add_mode == 'cat':
                    logits_cat = getattr(self, f"logits_cat{j}")[i]
                    cls_logits_cat = torch.cat((cls_logits_cat_up_list[::-1][i][:, j, :, :].unsqueeze(1),
                                                cls_logits_cat_down_list[i][:, j, :, :].unsqueeze(1)), dim=1)
                    logits_conv_feat_cls = logits_cat(cls_logits_cat)
                elif fu_add_mode == 'add':
                    #add_gate
                    if gate_mode == True:
                        cls_logits_cat = self.cls_logits_up_add1[i](cls_logits_cat_up_list[::-1][i][:, j, :, :].unsqueeze(1)) + self.cls_logits_down_add1[i](cls_logits_cat_down_list[i][:, j, :, :].unsqueeze(1))
                    else:
                        cls_logits_cat = cls_logits_cat_up_list[::-1][i][:, j, :, :].unsqueeze(1) + cls_logits_cat_down_list[i][:, j, :, :].unsqueeze(1)
                    logits_conv_feat_cls = cls_logits_cat
                #fusion attention
                if fu_attention_mode == True:
                    logits_conv_feat_cls = (1 + self.sigmoid(logits_conv_feat_cls)) * cls_logits_list[i][:, j, :, :].unsqueeze(1)

                logits_conv_feat_cls_list.append(logits_conv_feat_cls)
            logits_conv_feat = torch.cat(logits_conv_feat_cls_list, dim=1)

            cls_logits_cat_list.append(logits_conv_feat)

        cls_prob_list = cls_prob_cat_list
        cls_logits_list = cls_logits_cat_list

        for idx, (x, scale, stride) in enumerate(
                zip(feats, self.scales, self.prior_generator.strides)):

            b, c, h, w = x.shape
            anchor = anchor_list[idx]
            feat = feat_list[idx]
            reg_feat = reg_feat_list[idx]
            cls_logits = cls_logits_list[idx]
            cls_prob = cls_prob_list[idx]

            cls_score = sigmoid_geometric_mean(cls_logits, cls_prob)

            # reg prediction and alignment
            if self.anchor_type == 'anchor_free':
                reg_dist = scale(self.tood_reg(reg_feat).exp()).float()

                #per-class
                reg_bbox = []
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4*self.cls_out_channels)
                for i in range(self.cls_out_channels):
                    reg_bbox.append(distance2bbox(
                        self.anchor_center(anchor) / stride[0],
                        reg_dist[:, i*4:(i*4+4)], max_shape=torch.Tensor([h, w]).cuda()).reshape(b, h, w, 4).permute(0, 3, 1,2))  # (b, c, h, w)

            elif self.anchor_type == 'anchor_based':
                reg_dist = scale(self.tood_reg(reg_feat)).float()
                reg_dist = reg_dist.permute(0, 2, 3, 1).reshape(-1, 4)
                reg_bbox = self.bbox_coder.decode(anchor, reg_dist).reshape(
                    b, h, w, 4).permute(0, 3, 1, 2) / stride[0]
            else:
                raise NotImplementedError(
                    f'Unknown anchor type: {self.anchor_type}.'
                    f'Please use `anchor_free` or `anchor_based`.')
            reg_offset = self.reg_offset_module(feat)

            #per-class
            bbox_pred = []
            for i in range(self.cls_out_channels):
                bbox_pred.append(self.deform_sampling(reg_bbox[i].contiguous(),
                                                 reg_offset[:, i*8:(i*8+8)].contiguous()))

            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return tuple(cls_scores), tuple(bbox_preds)

    def deform_sampling(self, feat, offset):
        """Sampling the feature x according to offset.

        Args:
            feat (Tensor): Feature
            offset (Tensor): Spatial offset for for feature sampliing
        """
        # it is an equivalent implementation of bilinear interpolation
        b, c, h, w = feat.shape
        weight = feat.new_ones(c, 1, 1, 1)
        y = deform_conv2d(feat, offset, weight, 1, 0, 1, c, c)
        return y

    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights,
                    bbox_targets, alignment_metrics, stride, cls_idx, gt_bbox):  #per-class
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            alignment_metrics (Tensor): Alignment metrics with shape
                (N, num_total_anchors).
            stride (tuple[int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels).contiguous()
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)

        #per-class
        labels[labels == cls_idx] = 0
        labels[labels == self.cls_out_channels] = 1

        targets = labels if self.epoch < self.initial_epoch else (
            labels, alignment_metrics)

        cls_loss_func = self.initial_loss_cls \
            if self.epoch < self.initial_epoch else self.loss_cls

        #per-class
        loss_cls = cls_loss_func(
            cls_score[:, cls_idx].unsqueeze(1), targets, label_weights, avg_factor=1.0)
        bg_class_ind = 1

        pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]
            pos_anchors = anchors[pos_inds]

            pos_decode_bbox_pred = pos_bbox_pred
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]
            pos_decode_gt_bbox = gt_bbox / stride[0]

            #bbox refine
##            pos_bbox_preds_before_refine = bbox_preds_before_refine[pos_inds]
##            pos_decode_bbox_pred_before_refine = pos_bbox_preds_before_refine
            '''
            #bbox refine wh
            pos_decode_h_pred = (pos_decode_bbox_pred_before_refine[:, 3] - pos_decode_bbox_pred_before_refine[:, 1]).unsqueeze(1)
            pos_decode_w_pred = (pos_decode_bbox_pred_before_refine[:, 2] - pos_decode_bbox_pred_before_refine[:, 0]).unsqueeze(1)
            pos_decode_wh_pred = torch.cat((pos_decode_h_pred, pos_decode_w_pred), dim=1)

            pos_decode_h_targets = (pos_decode_bbox_targets[:, 3] - pos_decode_bbox_targets[:, 1]).unsqueeze(1)
            pos_decode_w_targets = (pos_decode_bbox_targets[:, 2] - pos_decode_bbox_targets[:, 0]).unsqueeze(1)
            pos_decode_wh_targets = torch.cat((pos_decode_h_targets, pos_decode_w_targets), dim=1)
            '''
            '''
            #loss scale
            pos_decode_h_pred = (pos_decode_bbox_pred[:, 3] - pos_decode_bbox_pred[:, 1]).unsqueeze(1)
            pos_decode_w_pred = (pos_decode_bbox_pred[:, 2] - pos_decode_bbox_pred[:, 0]).unsqueeze(1)
            pos_decode_wh_pred = torch.cat((pos_decode_h_pred, pos_decode_w_pred), dim=1)

            pos_decode_h_targets = (pos_decode_bbox_targets[:, 3] - pos_decode_bbox_targets[:, 1]).unsqueeze(1)
            pos_decode_w_targets = (pos_decode_bbox_targets[:, 2] - pos_decode_bbox_targets[:, 0]).unsqueeze(1)
            pos_decode_wh_targets = torch.cat((pos_decode_h_targets, pos_decode_w_targets), dim=1)
            '''
            '''
            #loss area
            pos_decode_area_pred = ((pos_decode_bbox_pred[:, 3] - pos_decode_bbox_pred[:, 1])*(pos_decode_bbox_pred[:, 2] - pos_decode_bbox_pred[:, 0])).unsqueeze(1)
            pos_decode_area_targets = ((pos_decode_bbox_targets[:, 3] - pos_decode_bbox_targets[:, 1])*(pos_decode_bbox_targets[:, 2] - pos_decode_bbox_targets[:, 0])).unsqueeze(1)
            '''
            # regression loss
            pos_bbox_weight = self.centerness_target(
                pos_anchors, pos_bbox_targets
            ) ##if self.epoch < self.initial_epoch else alignment_metrics[pos_inds]
            #equal weight
##            pos_bbox_weight = torch.ones(pos_anchors.shape[0]).cuda()

            #bbox refine wh
            '''
            pos_wh_offset_target_weight = torch.cat((pos_bbox_weight.unsqueeze(1), pos_bbox_weight.unsqueeze(1)), dim=1)
            loss_wh = self.loss_wh(
                pos_decode_wh_pred,
                pos_decode_wh_targets,
                pos_wh_offset_target_weight,
                avg_factor=1.0)
            '''
            '''
            #loss iof adaptive weight
            pos_bbox_weight_iogs = pos_bbox_weight * 2
            pos_decode_ious = bbox_overlaps(pos_decode_bbox_pred, pos_decode_bbox_targets, mode='iou', is_aligned=True, eps=1e-7)
            pos_bbox_weight_iogs = torch.where(pos_decode_ious<0.5, pos_bbox_weight_iogs, pos_bbox_weight)
            '''
            pos_bbox_weight_iogs = pos_bbox_weight

            '''
            #loss scale
            pos_wh_offset_target_weight = torch.cat((pos_bbox_weight.unsqueeze(1), pos_bbox_weight.unsqueeze(1)), dim=1)
            loss_scale = self.loss_scale(
                pos_decode_wh_pred,
                pos_decode_wh_targets,
                pos_wh_offset_target_weight,
                avg_factor=1.0)
            '''
            '''
            #loss area
            loss_area = self.loss_area(
                pos_decode_area_pred,
                pos_decode_area_targets,
                pos_bbox_weight,
                avg_factor=1.0)
            '''
            if self.iof_iog:
                #loss iof
                loss_iof = self.loss_iof(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets,
                    pos_decode_gt_bbox,
                    weight=pos_bbox_weight_iogs,
                    avg_factor=1.0)

                #loss iog
                loss_iog = self.loss_iog(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets,
                    pos_decode_gt_bbox,
                    weight=pos_bbox_weight,
                    avg_factor=1.0)

            #loss iou
            if self.epoch < 27:
                loss_bbox = self.loss_bbox(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets,
                    pos_decode_gt_bbox,
                    weight=pos_bbox_weight,
                    avg_factor=1.0)

                '''
                #bbox refine
                loss_wh = self.loss_wh(
                    pos_decode_bbox_pred_before_refine,
                    pos_decode_bbox_targets,
                    pos_decode_gt_bbox,
                    weight=pos_bbox_weight,
                    avg_factor=1.0)
                '''

            #loss iou
            else:
                loss_bbox = self.loss_bbox_last(
                    pos_decode_bbox_pred,
                    pos_decode_bbox_targets,
                    pos_decode_gt_bbox,
                    weight=pos_bbox_weight,
                    avg_factor=1.0)

        else:
            loss_bbox = bbox_pred.sum() * 0 #loss iou

            loss_iof = bbox_pred.sum() * 0    #loss iof
            loss_iog = bbox_pred.sum() * 0    #loss iog

            pos_bbox_weight = bbox_targets.new_tensor(0.)

        if self.iof_iog:
            return loss_cls, loss_bbox, loss_iof, loss_iog, alignment_metrics.sum(  #loss iof iog
            ), pos_bbox_weight.sum()
        else:
            return loss_cls, loss_bbox, alignment_metrics.sum(
            ), pos_bbox_weight.sum()

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             seg_scale,     #rail seg
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ], 1)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        #per-class
        flatten_bbox_preds = []
        for i in range(self.cls_out_channels):
            flatten_bbox_preds.append(torch.cat([
                bbox_pred[i].permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) * stride[0]
                for bbox_pred, stride in zip(bbox_preds,
                                             self.prior_generator.strides)
            ], 1))

        #rail seg
        flatten_seg_scale = torch.cat([
            seg.permute(0, 2, 3, 1).reshape(num_imgs, -1, 3)
            for seg, stride in zip(seg_scale,
                                         self.prior_generator.strides)
        ], 1)

        anchor_list_cls, labels_list_cls, label_weights_list_cls, bbox_targets_list_cls, alignment_metrics_list_cls = [], [], [], [], []
        for i in range(self.cls_out_channels):
            gt_labels_cls, gt_bboxes_cls = [], []
            for j in range(len(gt_labels)):
                idx = gt_labels[j] == i
                gt_labels_cls.append(gt_labels[j][idx])
                gt_bboxes_cls.append(gt_bboxes[j][idx, :])
            anchor_list, valid_flag_list = self.get_anchors(
                featmap_sizes, img_metas, device=device)
            cls_reg_targets = self.get_targets(
                flatten_cls_scores,
                flatten_bbox_preds[i],
                anchor_list,
                valid_flag_list,
                gt_bboxes_cls,
                img_metas,
                flatten_seg_scale,  #rail seg
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels_cls,
                label_channels=label_channels)
            (anchor_list2, labels_list, label_weights_list, bbox_targets_list,
             alignment_metrics_list) = cls_reg_targets
            anchor_list_cls.append(anchor_list2)
            labels_list_cls.append(labels_list)
            label_weights_list_cls.append(label_weights_list)
            bbox_targets_list_cls.append(bbox_targets_list)
            alignment_metrics_list_cls.append(alignment_metrics_list)

        losses_cls_list, losses_bbox_list, losses_wh_list, losses_scale_list, losses_iof_list, losses_iog_list = [], [], [], [], [], []
        for i in range(self.cls_out_channels):
            bbox_preds_cls = []
            for j in range(len(bbox_preds)):
                bbox_preds_cls.append(bbox_preds[j][i])

            if self.iof_iog:
                # per-class + loss iof + loss iog
                losses_cls, losses_bbox, losses_iof, losses_iog, cls_avg_factors, bbox_avg_factors = multi_apply(
                    self.loss_single,
                    anchor_list_cls[i],
                    cls_scores,
                    bbox_preds_cls,
                    labels_list_cls[i],
                    label_weights_list_cls[i],
                    bbox_targets_list_cls[i],
                    alignment_metrics_list_cls[i],
                    self.prior_generator.strides,
                    [i, i, i, i, i, i],
                    [gt_bboxes[0] for k in range(6)])
            else:
                losses_cls, losses_bbox, cls_avg_factors, bbox_avg_factors = multi_apply(
                    self.loss_single,
                    anchor_list_cls[i],
                    cls_scores,
                    bbox_preds_cls,
                    labels_list_cls[i],
                    label_weights_list_cls[i],
                    bbox_targets_list_cls[i],
                    alignment_metrics_list_cls[i],
                    self.prior_generator.strides,
                    [i, i, i, i, i, i],
                    [gt_bboxes[0] for k in range(6)])

            cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
            losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

            bbox_avg_factor = reduce_mean(
                sum(bbox_avg_factors)).clamp_(min=1).item()
            losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))   #loss iou

            losses_cls_list.append(losses_cls)
            losses_bbox_list.append(losses_bbox)    #loss iou

            if self.iof_iog:
                #per-class + loss iof
                losses_iof = list(map(lambda x: x / bbox_avg_factor, losses_iof))
                losses_iof_list.append(losses_iof)
                #per-class + loss iog
                losses_iog = list(map(lambda x: x / bbox_avg_factor, losses_iog))
                losses_iog_list.append(losses_iog)

        losses_cls_sum, losses_bbox_sum, losses_wh_sum, losses_scale_sum, losses_iof_sum, losses_iog_sum = [], [], [], [], [], []
        for i in range(len(losses_cls)):
            if self.cls_out_channels == 3:
                losses_cls_sum.append(losses_cls_list[0][i] + losses_cls_list[1][i] + losses_cls_list[2][i])
                #per-class + loss iou
                losses_bbox_sum.append(losses_bbox_list[0][i] + losses_bbox_list[1][i] + losses_bbox_list[2][i])
                if self.iof_iog:
                    #per-class + loss iof
                    losses_iof_sum.append(losses_iof_list[0][i] + losses_iof_list[1][i] + losses_iof_list[2][i])
                    # per-class + loss iog
                    losses_iog_sum.append(losses_iog_list[0][i] + losses_iog_list[1][i] + losses_iog_list[2][i])
            elif self.cls_out_channels == 1:
                losses_cls_sum.append(losses_cls_list[0][i])
                #loss iou
                losses_bbox_sum.append(losses_bbox_list[0][i])
                #loss iof
                losses_iof_sum.append(losses_iof_list[0][i])
                #loss iog
                losses_iog_sum.append(losses_iog_list[0][i])

        if self.iof_iog:
            #per-class + loss iof + loss iog
            return dict(loss_cls=losses_cls_sum, loss_bbox=losses_bbox_sum, loss_iof=losses_iof_sum, loss_iog=losses_iog_sum)
        else:
            #per-class
            return dict(loss_cls=losses_cls_sum, loss_bbox=losses_bbox_sum)

    def _get_bboxes_single(self,
                           cls_score_list,
                           bbox_pred_list,
                           score_factor_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (tl_x, tl_y, br_x, br_y) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """

        cfg = self.test_cfg if cfg is None else cfg
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for cls_score, bbox_pred, priors, stride in zip(
                cls_score_list, bbox_pred_list, mlvl_priors,
                self.prior_generator.strides):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4) * stride[0]
##            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            scores = cls_score.permute(1, 2, 0).reshape(-1, 1)   #per-class

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bboxes = filtered_results['bbox_pred']

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        return self._bbox_post_process(mlvl_scores, mlvl_labels, mlvl_bboxes,
                                       img_meta['scale_factor'], cfg, rescale,
                                       with_nms, None, **kwargs)

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    seg_scale,  #rail seg
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores (Tensor): Classification predictions of images,
                a 3D-Tensor with shape [num_imgs, num_priors, num_classes].
            bbox_preds (Tensor): Decoded bboxes predictions of one image,
                a 3D-Tensor with shape [num_imgs, num_priors, 4] in [tl_x,
                tl_y, br_x, br_y] format.
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: a tuple containing learning targets.

                - anchors_list (list[list[Tensor]]): Anchors of each level.
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - norm_alignment_metrics_list (list[Tensor]): Normalized
                  alignment metrics of each level.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        # anchor_list: list(b * [-1, 4])

        if self.epoch < self.initial_epoch:
            (all_anchors, all_labels, all_label_weights, all_bbox_targets,
             all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
                 super()._get_target_single,
                 anchor_list,
                 valid_flag_list,
                 num_level_anchors_list,
                 gt_bboxes_list,
                 gt_bboxes_ignore_list,
                 gt_labels_list,
                 img_metas,
                 seg_scale,     #rail seg
                 label_channels=label_channels,
                 unmap_outputs=unmap_outputs)
            all_assign_metrics = [
                weight[..., 0] for weight in all_bbox_weights
            ]
        else:

            (all_anchors, all_labels, all_label_weights, all_bbox_targets,
             all_assign_metrics, all_bbox_weights) = multi_apply(
                 self._get_target_single,
                 cls_scores,
                 bbox_preds,
                 anchor_list,
                 valid_flag_list,
                 gt_bboxes_list,
                 gt_bboxes_ignore_list,
                 gt_labels_list,
                 img_metas,
                 seg_scale,  #rail seg
                 num_level_anchors_list,    #atss assigner
                 label_channels=label_channels,
                 unmap_outputs=unmap_outputs)
            '''
            #atss assigner
            bbox_preds_list = [bbox_preds[i, :, :] for i in range(bbox_preds.shape[0])]
            (all_anchors, all_labels, all_label_weights, all_bbox_targets,
             all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
                super()._get_target_single,
                bbox_preds_list,
                valid_flag_list,
                num_level_anchors_list,
                gt_bboxes_list,
                gt_bboxes_ignore_list,
                gt_labels_list,
                img_metas,
                seg_scale,  # rail seg
                label_channels=label_channels,
                unmap_outputs=unmap_outputs)
            all_assign_metrics = [weight[..., 0] for weight in all_bbox_weights] #without weight
            '''
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None

        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        norm_alignment_metrics_list = images_to_levels(all_assign_metrics,
                                                       num_level_anchors)

        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, norm_alignment_metrics_list)

    def _get_target_single(self,
                           cls_scores,
                           bbox_preds,
                           flat_anchors,
                           valid_flags,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           seg_scale,  # rail seg
                           num_level_anchors,
                           label_channels=1,
                           unmap_outputs=True):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (list(Tensor)): Box scores for each image.
            bbox_preds (list(Tensor)): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                norm_alignment_metrics (Tensor): Normalized alignment metrics
                    of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        #atss assigner
        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)

        assign_result = self.alignment_assigner.assign(
            cls_scores[inside_flags, :], bbox_preds[inside_flags, :], anchors, gt_bboxes, num_level_anchors_inside, #atss assigner
            gt_bboxes_ignore, gt_labels, self.alpha, self.beta, self.epoch)
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        norm_alignment_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
##            label_weights[neg_inds] = 1
            label_weights[neg_inds] = 1 / float(self.cls_out_channels) #per-class
        '''
        #rail seg
        seg_valid = seg_scale[:, 0][inside_flags]
        rail = seg_valid * 0.0
        unrelated = (1 - seg_valid) * 1.0
        label_weights_rail = label_weights * rail
        label_weights_unrelated = label_weights * unrelated
        label_weights = label_weights_rail + label_weights_unrelated
        label_weights[pos_inds] = 1.5
        '''
        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            norm_alignment_metrics = unmap(norm_alignment_metrics,
                                           num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets,
                norm_alignment_metrics, bbox_weights)

    def star_dcn_offset(self, bbox_pred, gradient_mul, stride):
        """Compute the star deformable conv offsets.

        Args:
            bbox_pred (Tensor): Predicted bbox distance offsets (l, r, t, b).
            gradient_mul (float): Gradient multiplier.
            stride (int): The corresponding stride for feature maps,
                used to project the bbox onto the feature map.

        Returns:
            dcn_offsets (Tensor): The offsets for deformable convolution.
        """
        dcn_base_offset = self.dcn_base_offset.type_as(bbox_pred)
        bbox_pred_grad_mul = (1 - gradient_mul) * bbox_pred.detach() + \
            gradient_mul * bbox_pred
        # map to the feature map scale
        bbox_pred_grad_mul = bbox_pred_grad_mul / stride
        N, C, H, W = bbox_pred.size()

        x1 = bbox_pred_grad_mul[:, 0, :, :]
        y1 = bbox_pred_grad_mul[:, 1, :, :]
        x2 = bbox_pred_grad_mul[:, 2, :, :]
        y2 = bbox_pred_grad_mul[:, 3, :, :]
        bbox_pred_grad_mul_offset = bbox_pred.new_zeros(
            N, 2 * self.num_dconv_points, H, W)
        bbox_pred_grad_mul_offset[:, 0, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 1, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 2, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 4, :, :] = -1.0 * y1  # -y1
        bbox_pred_grad_mul_offset[:, 5, :, :] = x2  # x2
        bbox_pred_grad_mul_offset[:, 7, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 11, :, :] = x2  # x2
        bbox_pred_grad_mul_offset[:, 12, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 13, :, :] = -1.0 * x1  # -x1
        bbox_pred_grad_mul_offset[:, 14, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 16, :, :] = y2  # y2
        bbox_pred_grad_mul_offset[:, 17, :, :] = x2  # x2
        dcn_offset = bbox_pred_grad_mul_offset - dcn_base_offset

        return dcn_offset

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside