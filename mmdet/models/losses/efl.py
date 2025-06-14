# Import from third library
import torch

# Import from pod
from .eqfl import GeneralizedCrossEntropyLoss
from ..builder import LOSSES

import torch.nn.functional as F
import torch.distributed as dist

try:
    import spring.linklink as link
except:   # noqa
    link = None

class DistBackend():
    def __init__(self):
        self.backend = 'dist'

DIST_BACKEND = DistBackend()

def allreduce(*args, **kwargs):
    if DIST_BACKEND.backend == 'linklink':
        return link.allreduce(*args, **kwargs)
    elif DIST_BACKEND.backend == 'dist':
        return dist.all_reduce(*args, **kwargs)
    else:
        raise NotImplementedError

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

@LOSSES.register_module()
class EqualizedFocalLoss(GeneralizedCrossEntropyLoss):
    def __init__(self,
                 name='equalized_focal_loss',
                 reduction='mean',
                 use_sigmoid=True,
                 activated=True,
                 loss_weight=1.0,
                 ignore_index=-1,
                 num_classes=4,
                 focal_gamma=2.0,
                 focal_alpha=0.25,
                 scale_factor=4.0,
                 fpn_levels=5):
        activation_type = 'sigmoid'
        GeneralizedCrossEntropyLoss.__init__(self,
                                             name=name,
                                             reduction=reduction,
                                             loss_weight=loss_weight,
                                             activation_type=activation_type,
                                             ignore_index=ignore_index)

        # cfg for focal loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

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


##        logger.info(f"build EqualizedFocalLoss, focal_alpha: {focal_alpha}, focal_gamma: {focal_gamma}, \
##                    scale_factor: {scale_factor}")

    def forward(self, input, target, reduction, normalizer=None):
        self.n_c = input.shape[-1]
        self.input = input.reshape(-1, self.n_c)
        self.target = target.reshape(-1)
        self.n_i, _ = self.input.size()
        self.eps = 1e-6

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c + 1)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target[:, :self.n_c]

        expand_target = expand_label(self.input, self.target)
        sample_mask = (self.target != self.ignore_index)

        inputs = self.input[sample_mask]
        targets = expand_target[sample_mask]

        self.cache_mask.append(sample_mask.unsqueeze(1))
        self.cache_target.append(expand_target)

        pred = inputs    #pred.sigmoid(inputs)
        pred_t = pred * targets + (1 - pred) * (1 - targets)

        # ce_loss
        ce_loss = -torch.log(pred_t + self.eps)
        '''
##        output = _reduce(ce_loss, reduction, normalizer=normalizer)
##        grad_i = torch.autograd.grad(outputs=torch.exp(output), inputs=pred)[0]  # 求导
        grad_i = torch.abs(targets * (pred - 1) + (1 - targets) * pred)
        grad_i = grad_i.gather(1, targets.long())  # 每个类对应的梯度
        '''
        map_val = 1 - self.pos_neg.detach()
        dy_gamma = self.focal_gamma + self.scale_factor * map_val
        # focusing factor
        ff = dy_gamma.view(1, -1).expand(self.n_i, self.n_c)[sample_mask]
        # weighting factor
        wf = ff / self.focal_gamma

        cls_loss = ce_loss * torch.pow((1 - pred_t), ff.detach()) * wf.detach()

        # to avoid an OOM error
        # torch.cuda.empty_cache()

        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
            cls_loss = alpha_t * cls_loss
        '''
        pos_grad_i = torch.sum(grad_i * targets * wf * self.focal_alpha, dim=0)
        neg_grad_i = torch.sum(grad_i * (1 - targets) * wf * (1 - self.focal_alpha), dim=0)
        neg_grad_i += 1e-9  # 防止除数为0
##        grad_i = pos_grad_i / neg_grad_i
##        grad_i = torch.clamp(grad_i, min=0, max=1)  # 裁剪梯度
        self.pos_grad += pos_grad_i
        self.neg_grad += neg_grad_i
        self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1)
        '''
        if normalizer is None:
            normalizer = 1.0
        '''
        output = _reduce(cls_loss, reduction, normalizer=normalizer)
        grad_i = torch.autograd.grad(outputs=output, inputs=pred)[0]  # 求导
        grad_i = grad_i.gather(1, targets.long())  # 每个类对应的梯度
        self.collect_grad(grad_i)
        '''
        return _reduce(cls_loss, reduction, normalizer=normalizer)

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

##        allreduce(pos_grad)
##        allreduce(neg_grad)

            self.pos_grad += pos_grad
            self.neg_grad += neg_grad
            self.pos_neg = torch.clamp(self.pos_grad / (self.neg_grad + 1e-10), min=0, max=1)

            self.grad_buffer = []

            self.cache_target = []
            self.cache_mask = []
