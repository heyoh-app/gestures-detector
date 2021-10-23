import math
import os

import numpy as np
import torch
from torch.utils import data

from .augmentations import Transforms
from .utils import read_img, clip_boxes, draw_gaussian, gaussian_radius, resize_boxes


def get_filename(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]


class Dataset(data.Dataset):

    def __init__(
            self,
            files,
            images_folder,
            subclasses,
            radius_weights,
            output_stride,
            scope="weak",
            size_transform="center",
            size=256,
    ):
        self.files = files
        self.size = size
        self.scope = scope
        self.size_transform = size_transform
        self.transforms = Transforms(
            size=self.size,
            scope=self.scope,
            size_transform=self.size_transform
        )
        self.images_folder = images_folder
        self.subclasses = subclasses
        self.num_classes = len(subclasses)
        self.num_subclasses = sum(subclasses)
        self.radius_weights = radius_weights
        self.out_stride = output_stride
        self.out_size = self.size // self.out_stride

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_name = self.files[index][0]
        bboxes = self.files[index][1]

        # read image
        image_path = os.path.join(self.images_folder, image_name)
        image = read_img(image_path)
        h, w, _ = image.shape

        # albumentations only supports boxes which are strictly inside the image
        bboxes = clip_boxes(bboxes, h, w)

        # augment image and masks
        data = {"image": image, "bboxes": bboxes}
        image, bboxes = self.transforms.get_augmented(data)

        # init masks for keypoints, sizes, sides
        mask_kpoint = np.zeros((self.num_subclasses, self.out_size, self.out_size))
        mask_size = np.zeros((1, self.out_size, self.out_size), dtype=np.float32)

        # we are going to use 0 for left, 1 for right, 0.5 otherwise
        mask_side = 0.5 * np.ones((1, self.out_size, self.out_size), dtype=np.float32)

        # resize bboxes according to out_stride
        bboxes = resize_boxes(bboxes, 1 / self.out_stride)
        for bbox in bboxes:
            # heatmaps
            obj_center = [np.mean(bbox[:4:2]), np.mean(bbox[1:4:2])]
            obj_center = list(map(int, obj_center))
            obj_w = bbox[2] - bbox[0]
            obj_h = bbox[3] - bbox[1]
            obj_class, obj_subclass, obj_side = bbox[4:7].astype(int)
            obj_cat = sum(self.subclasses[:obj_class]) + obj_subclass

            # object side
            radius_weight = self.radius_weights[obj_class]  # adjust gaussian radius according to class_weights
            radius = math.ceil(radius_weight * gaussian_radius([obj_h, obj_w]))
            heatmap = draw_gaussian(np.zeros((self.out_size, self.out_size)), obj_center, radius)
            mask_kpoint[obj_cat] = np.maximum(mask_kpoint[obj_cat], heatmap)

            # object side
            if obj_side != -1:  # we use "-1" for classes without 'side' attribute
                mask_side[0, obj_center[1] - 1: obj_center[1] + 2, obj_center[0] - 1: obj_center[0] + 2] = obj_side

            # object size
            mask_size[0, obj_center[1] - 1: obj_center[1] + 2,
            obj_center[0] - 1: obj_center[0] + 2] = obj_h / self.out_size

        image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
        mask_kpoint, mask_side, mask_size = map(torch.from_numpy, [mask_kpoint, mask_side, mask_size])

        return image, mask_kpoint, mask_side, mask_size
