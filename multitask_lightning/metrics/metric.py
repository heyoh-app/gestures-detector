import torch
import numpy as np
from mean_average_precision import MetricBuilder
from .utils import masks_to_bboxes
from functools import partial


class MeanAveragePrecision:
    def __init__(
            self,
            num_classes: int,
            out_img_size: int = 64,
            threshold_iou: float = 0.5,
            threshold_kpoint_prob: float = 0.4,
            max_bbox_per_img: int = 5
    ):
        self.threshold_iou = threshold_iou
        self.map_metric = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=num_classes)
        self.masks_to_bboxes = partial(
            masks_to_bboxes,
            num_classes=num_classes,
            out_size=out_img_size,
            threshold=threshold_kpoint_prob,
            max_bbox_per_img=max_bbox_per_img
        )

    def update(self, predict: torch.Tensor, gt: torch.Tensor):

        bboxes_gt_batch = self.masks_to_bboxes(gt)
        bboxes_predict_batch = self.masks_to_bboxes(predict, is_predict=True)

        for bboxes_predict, bboxes_gt in zip(bboxes_predict_batch, bboxes_gt_batch):
            self.map_metric.add(np.array(bboxes_predict), np.array(bboxes_gt))

    def pascal_map_value(self, reset: bool = True):
        pascal_map = self.map_metric.value(iou_thresholds=self.threshold_iou)['mAP']
        if reset:
            self.map_metric.reset()
        return pascal_map

