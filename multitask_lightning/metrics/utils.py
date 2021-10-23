import torch
import numpy as np
import sys
sys.path.append("../")
from utils.train_utils import nms

def postprocess(tensor, tensor_type, threshold=None):
    with torch.no_grad():
        if tensor_type == "kpoint":
            threshold = torch.nn.Threshold(threshold, 0)
            tensor = nms(threshold(torch.sigmoid(tensor)))
        elif tensor_type == "size":
            tensor = torch.sigmoid(tensor)
    return tensor

def masks_to_bboxes(masks, num_classes, max_bbox_per_img: int, threshold: float, out_size: int, is_predict: bool = False):
    kpoint, side, size = torch.split(masks, [num_classes, 1, 1], 1) if is_predict else masks

    if is_predict:
        kpoint = postprocess(kpoint, tensor_type="kpoint", threshold=threshold)
        size = postprocess(size, tensor_type="size")
    else:
        kpoint = torch.eq(kpoint, 1.).float()

    binary_mask = torch.gt(kpoint, 0)
    coords = binary_mask.nonzero().cpu().detach().numpy()   # get list of object center coords [[B,C,H,W], ...]
    probs = kpoint.masked_select(binary_mask).cpu().detach().numpy()   # get list of probabilities of each object
    heights = (binary_mask * size).masked_select(binary_mask).cpu().detach().numpy()   # get list of heights of each object

    # [x top-left, y top-left, x bottom-right, y bottom-right, class, probability, img idx in batch]
    bboxes = np.zeros((coords.shape[0], 7))
    bboxes[:, 0] = coords[:, 3] - 0.5 * heights * out_size
    bboxes[:, 1] = coords[:, 2] - 0.5 * heights * out_size
    bboxes[:, 2] = coords[:, 3] + 0.5 * heights * out_size
    bboxes[:, 3] = coords[:, 2] + 0.5 * heights * out_size
    bboxes[:, 4] = coords[:, 1]
    bboxes[:, 5] = probs
    bboxes[:, 6] = coords[:, 0]

    bboxes_batch_list = []
    for b in range(kpoint.shape[0]):   # batch size
        bboxes_batch = bboxes[bboxes[:, -1] == b][:, :-1]

        if is_predict:  # filter top k boxes
            bboxes_batch = bboxes_batch[bboxes_batch[:, -1].argsort()][::-1]
            bboxes_batch = bboxes_batch[:max_bbox_per_img]
        else:
            # [x top-left, y top-left, x bottom-right, y bottom-right, class, difficult, crowd]
            bboxes_batch = np.hstack([
                bboxes_batch[:, :5],
                np.zeros((bboxes_batch.shape[0], 2))
            ])

        bboxes_batch_list.append(bboxes_batch)

    return bboxes_batch_list
