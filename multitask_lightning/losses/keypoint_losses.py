from __future__ import print_function, division
import torch
from torch import nn


def _neg_loss(pred, gt):

    """Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    """

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class KpointFocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self, clamp=0.001, weight=1.):
        super(KpointFocalLoss, self).__init__()
        self.neg_loss = _neg_loss
        self.clamp = clamp
        self.weight = weight

    def forward(self, out, target):
        out = torch.sigmoid(out)
        out = torch.clamp(out, min=self.clamp, max=1-self.clamp)  # for loss stabilization
        return self.weight * self.neg_loss(out, target)
