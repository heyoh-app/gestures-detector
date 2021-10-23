import torch
from torch import nn
from pytorch_toolbelt.losses import BinaryFocalLoss


def _reg_l1_loss(regr, gt, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()

    regr = regr * mask
    gt = gt * mask

    regr_loss = nn.functional.smooth_l1_loss(regr, gt, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_l2_loss(regr, gt, mask):
    ''' L2 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()

    regr = regr * mask
    gt = gt * mask

    loss_function = nn.MSELoss(reduction='sum')

    regr_loss = loss_function(regr, gt)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


LOSSES = {
    "l1": _reg_l1_loss,
    "l2": _reg_l2_loss
}
ACTIVATIONS = {
    "sigmoid": nn.Sigmoid(),
    "softsign": nn.Softsign(),
    "tanh": nn.Tanh()
}


class RegrLoss(nn.Module):

    def __init__(self, distance="l1", activation="sigmoid", weight=1.0):
        super(RegrLoss, self).__init__()
        self.loss = LOSSES[distance]
        self.activation = ACTIVATIONS[activation]
        self.weight = weight

    def forward(self, out, target):
        mask = torch.gt(target, 0).float()
        out = self.activation(out)

        return self.weight * self.loss(out, target, mask)


class MaskedFocal(nn.Module):

    def __init__(self, activation="sigmoid", weight=1.):
        super(MaskedFocal, self).__init__()
        self.loss = BinaryFocalLoss(alpha=None, gamma=2)
        self.activation = ACTIVATIONS[activation]
        self.weight = weight

    def forward(self, out, target):
        out = self.activation(out)

        mask = ~target.eq(0.5)      # 0 for left, 1 for right, 0.5 otherwise

        pred = out.masked_select(mask)
        gt = target.masked_select(mask)

        return self.weight * self.loss(pred, gt)
