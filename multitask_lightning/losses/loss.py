from .keypoint_losses import KpointFocalLoss
from .aux_losses import RegrLoss, MaskedFocal


def get_loss(loss):
    loss_name = loss["name"]
    params = loss["params"]

    if loss_name == "kpoint_focal":
        loss = KpointFocalLoss(**params)
    elif loss_name == "masked_focal":
        loss = MaskedFocal()
    elif loss_name == "regr_loss":
        loss = RegrLoss(**params)
    else:
        raise ValueError("Loss [%s] not recognized." % loss_name)

    return loss
