# Import from third library
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.modules.loss import _Loss
from ..builder import LOSSES

# Import from pod
##from eod.utils.general.log_helper import default_logger as logger
try:
    import spring.linklink as link
except:   # noqa
    link = None

__all__ = ['EqualizedQualityFocalLoss']

class DistBackend():
    def __init__(self):
        self.backend = 'linklink'

DIST_BACKEND = DistBackend()

def allreduce(*args, **kwargs):
    if DIST_BACKEND.backend == 'linklink':
        return link.allreduce(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.all_reduce(*args, **kwargs)
    else:
        raise NotImplementedError

def dynamic_normalizer(input, target, alpha, gamma):
    def reduce_(tensor, gamma):
        return tensor.pow(gamma).sum()
    target = target.reshape(-1).long()
    input_p = input.detach().sigmoid()
    pos_mask = torch.nonzero(target >= 1).squeeze()
    valid_mask = torch.nonzero(target >= 0).squeeze()
    pos_normalizer = reduce_((1 - input_p[pos_mask, target[pos_mask] - 1]), gamma)
    neg_normalizer = reduce_(input_p[valid_mask], gamma) - reduce_(input_p[pos_mask, target[pos_mask] - 1], gamma)
    pos_normalizer *= alpha
    neg_normalizer *= 1 - alpha
    normalizer = torch.clamp(pos_normalizer + neg_normalizer, min=1)
    return normalizer


def _reduce(loss, reduction, **kwargs):
    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        normalizer = loss.numel()
        if kwargs.get('normalizer', None):
            normalizer = kwargs['normalizer']
        ret = loss.sum() / normalizer
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        raise ValueError(reduction + ' is not valid')
    return ret

class BaseLoss(_Loss):
    # do not use syntax like `super(xxx, self).__init__,
    # which will cause infinited recursion while using class decorator`
    def __init__(self,
                 name='base',
                 reduction='none',
                 loss_weight=1.0):
        r"""
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
        """
        _Loss.__init__(self, reduction=reduction)
        self.loss_weight = loss_weight
        self.name = name

    def __call__(self, input, target, weight=None, avg_factor=None, reduction_override=None, normalizer_override=None, **kwargs):
        r"""
        Arguments:
            - input (:obj:`Tensor`)
            - reduction (:obj:`Tensor`)
            - reduction_override (:obj:`str`): choice of 'none', 'mean', 'sum', override the reduction type
            defined in __init__ function

            - normalizer_override (:obj:`float`): override the normalizer when reduction is 'mean'
        """
        reduction = reduction_override if reduction_override else self.reduction
        assert (normalizer_override is None or reduction == 'mean'), \
            f'normalizer is not allowed when reduction is {reduction}'
        loss = _Loss.__call__(self, input, target, reduction, normalizer=normalizer_override, **kwargs)

        return loss * self.loss_weight

    def forward(self, input, target, reduction, normalizer=None, **kwargs):
        raise NotImplementedError

@LOSSES.register_module()
class GeneralizedCrossEntropyLoss(BaseLoss):
    def __init__(self,
                 name='generalized_cross_entropy_loss',
                 reduction='none',
                 loss_weight=1.0,
                 activation_type='softmax',
                 ignore_index=-1,):
        BaseLoss.__init__(self,
                          name=name,
                          reduction=reduction,
                          loss_weight=loss_weight)
        self.activation_type = activation_type
        self.ignore_index = ignore_index

@LOSSES.register_module()
class EqualizedQualityFocalLoss(GeneralizedCrossEntropyLoss):
    """
    Quality focal loss: https://arxiv.org/abs/2006.04388,
    """
    def __init__(self,
                 name='equalized_quality_focal_loss',
                 reduction='mean',
                 use_sigmoid=True,
                 activated=True,
                 loss_weight=1.0,
                 ignore_index=-1,
                 num_classes=4,
                 focal_gamma=2.0,
                 scale_factor=4.0,
                 fpn_levels=5,
                 dynamic_normalizer=False):
        """
        Arguments:
            - name (:obj:`str`): name of the loss function
            - reduction (:obj:`str`): reduction type, choice of mean, none, sum
            - loss_weight (:obj:`float`): loss weight
        """
        activation_type = 'sigmoid'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)

        self.dynamic_normalizer = dynamic_normalizer
        assert ignore_index == -1, 'only -1 is allowed for ignore index'

        # cfg for focal loss
        self.focal_gamma = focal_gamma

        # ignore bg class and ignore idx
        self.num_classes = num_classes - 1

        # cfg for efl loss
        self.scale_factor = scale_factor
        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes))
        self.register_buffer('neg_grad', torch.zeros(self.num_classes))
        self.register_buffer('pos_neg', torch.ones(self.num_classes))

        # grad collect
        self.grad_buffer = []
        self.fpn_levels = fpn_levels

        self.cache_mask = []
        self.cache_target = []

##        logger.info(f"build EqualizedQualityFocalLoss, focal_gamma: {focal_gamma}, scale_factor: {scale_factor}")

    def forward(self, input, target, reduction, normalizer=None, scores=None):
        """
        Arguments:
            - input (FloatTenosor): [[M, N,]C]
            - target (LongTenosor): [[M, N]]
        """
        assert reduction != 'none', 'Not Supported none reduction yet'
        target, scores = target
        self.n_c = input.shape[-1]
        self.input = input.reshape(-1, self.n_c)
        self.target = target.reshape(-1)
        self.n_i, _ = self.input.size()
        scores = scores.reshape(-1)

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        expand_target = expand_label(self.input, self.target)
        sample_mask = (self.target != self.ignore_index)

        inputs = self.input[sample_mask]
        scores = scores[sample_mask]

        # cache for gradient collector
        self.cache_mask.append(sample_mask.unsqueeze(1))
        self.cache_target.append(expand_target)

        # normlizer
        normalizer = 1.0 if normalizer is None else normalizer
        normalizer = torch.Tensor([normalizer]).type_as(inputs).to(inputs.device)
        if self.dynamic_normalizer:
            normalizer = dynamic_normalizer(inputs, self.target[sample_mask], 0.5, self.focal_gamma)

        pred = inputs   #torch.sigmoid(inputs)
        map_val = 1 - self.pos_neg.detach()
        dy_gamma = self.focal_gamma + self.scale_factor * map_val
        # focusing factor
        ff = dy_gamma.view(1, -1).expand(self.n_i, self.n_c)[sample_mask]
        # weighting factor
        wf = ff / self.focal_gamma

        # loss of negative samples
        loss = F.binary_cross_entropy_with_logits(inputs,
                                                  inputs.new_zeros(inputs.shape),
                                                  reduction='none') * pred.pow(ff.detach())
        # loss of positive samples
        '''
        pos_inds = torch.where(self.target[sample_mask] != 0)[0]
        pos_target = self.target[sample_mask][pos_inds].long() - 1
        quality_factor = scores[pos_inds] - pred[pos_inds, pos_target]
        loss[pos_inds, pos_target] = F.binary_cross_entropy_with_logits(inputs[pos_inds, pos_target],
                                                                        scores[pos_inds],
                                                                        reduction='none') * quality_factor.abs().pow(ff.detach()[pos_inds, pos_target]) # noqa

        '''
        bg_class_ind = pred.size(1)
        pos = ((self.target[sample_mask] >= 0) & (self.target[sample_mask] < bg_class_ind)).nonzero().squeeze(1)
        pos_target = self.target[sample_mask][pos].long()
        quality_factor = scores[pos] - pred[pos, pos_target]
        loss[pos, pos_target] = F.binary_cross_entropy_with_logits(inputs[pos, pos_target],
                                                                        scores[pos],
                                                                        reduction='none') * quality_factor.abs().pow(ff.detach()[pos, pos_target]) # noqa

        loss = loss * wf.detach()
        return _reduce(loss, reduction=reduction, normalizer=normalizer)

    def collect_grad(self, grad_in):
        bs = grad_in.shape[0]
        grad_append = grad_in.detach().permute(0, 2, 3, 1).reshape(bs, -1, self.num_classes)
        self.grad_buffer.append(grad_append)
        '''
        while (len(self.grad_buffer) >= 2) and (self.grad_buffer[-1].shape == self.grad_buffer[-2].shape):
            del self.grad_buffer[-1]
        if (len(self.grad_buffer) >= 2) and (self.grad_buffer[0].shape[1] > self.grad_buffer[1].shape[1]):
            del self.grad_buffer[0]
        '''
        if len(self.grad_buffer) == self.fpn_levels:
            self.cache_target = torch.cat(self.cache_target, dim=0)
            self.cache_mask = torch.cat(self.cache_mask, dim=0).squeeze(1)

            target = self.cache_target[self.cache_mask]
            grad = torch.cat(self.grad_buffer[::-1], dim=1).reshape(-1, self.num_classes)

            grad = torch.abs(grad)[self.cache_mask]
            pos_grad = torch.sum(grad * target, dim=0)
            neg_grad = torch.sum(grad * (1 - target), dim=0)

##            allreduce(pos_grad)
##            allreduce(neg_grad)

            self.pos_grad += pos_grad
            self.neg_grad += neg_grad
            self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1)

            self.grad_buffer = []

            self.cache_target = []
            self.cache_mask = []

