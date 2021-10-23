from typing import Tuple
import torch

import numbers
import numpy as np
from PIL import Image
from torchvision.transforms.transforms import ToTensor
from multitask_lightning.utils.filter import OneEuroFilter
from coremltools.models import model as coreml
from torchvision import transforms
import torchvision.transforms.functional as F

FACE_MASK_CHANNELS = (0, 2)
PALMS_CHANNELS = 2
output_stride = 2


class PadToSize(object):
    def __init__(self, size=(128, 256), fill=0, padding_mode="constant"):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]

        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        w_pad = self.size[1] - img.size[0]
        h_pad = self.size[0] - img.size[1]

        return F.pad(
            img,
            (w_pad // 2, h_pad // 2, w_pad // 2, h_pad // 2),
            self.fill,
            self.padding_mode,
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(padding={0}, fill={1}, padding_mode={2})".format(
                self.fill, self.padding_mode
            )
        )


def preprocess(img: np.ndarray) -> tuple:
    inference_pipeline = transforms.Compose(
        [
            transforms.Resize(size=(128, 228)),
            PadToSize(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449, 0.449, 0.449], std=[0.226, 0.226, 0.226]),
        ]
    )

    model_input = inference_pipeline(Image.fromarray(img))
    model_input = model_input.unsqueeze(0)

    return model_input, max(img.shape) / max(model_input.shape)


def predict_with_jit(
    model: torch.jit.TracedModule, img: np.ndarray
) -> Tuple[torch.Tensor, float]:
    model_input, scale = preprocess(img)
    output = model(model_input)
    return output, scale


def predict_with_coreml(
    model: coreml.MLModel, img: np.ndarray
) -> Tuple[torch.Tensor, float]:

    inference_pipeline = transforms.Compose(
        [transforms.Resize(size=(128, 228)), PadToSize()]
    )
    model_input = inference_pipeline(Image.fromarray(img))
    output = model.predict({"input": model_input})

    return (
        output["probs"],
        output["coords"],
        output["size_width"],
        output["size_height"],
        output["side_squeezed"],
    ), max(img.shape) / max(np.array(model_input).shape)


def predict(img, model, debug: bool):
    if debug:
        output, scale = predict_with_jit(model, img)
    else:
        output, scale = predict_with_coreml(model, img)

    predictions = []
    probs, points, size_width, size_height, side = output

    for idx in range(len(points)):
        predicted_class, y, x = points[idx]
        prob = probs[idx]
        x = int(x * scale * output_stride)
        y = int(y * scale * output_stride)

        width = int(size_width[idx] * 128) * scale
        height = int(size_height[idx] * 128) * scale

        x_lt = x - width / 2
        y_lt = y - height / 2
        x_rb = x + width / 2
        y_rb = y + height / 2

        predictions.append(
            (
                prob,
                int(predicted_class),
                x,
                y,
                float(side[idx]),
                (x_lt, y_lt, x_rb, y_rb),
                (width, height),
            )
        )

    return predictions


class Track:

    """Tracks hand, smoothes predictions and propagates coordinates in case of missing detection"""

    def __init__(self, hand_side, max_propagate=5, min_num_predictions=10):
        self.history = []
        self.propagate_count = 0
        self.max_propagate = max_propagate
        self.hand_side = hand_side
        self.min_num_predictions = min_num_predictions
        self._init_filters()

    def _init_filters(self):

        self.smoothing_coord_config = {
            "freq": 120,
            "mincutoff": 0.9,
            "beta": 0.05,
            "dcutoff": 1.0,
        }

        self.smoothing_size_config = {
            "freq": 120,
            "mincutoff": 0.3,
            "beta": 0.005,
            "dcutoff": 1.0,
        }

        self.x_filter = OneEuroFilter(**self.smoothing_coord_config)
        self.y_filter = OneEuroFilter(**self.smoothing_coord_config)
        self.w_filter = OneEuroFilter(**self.smoothing_size_config)
        self.h_filter = OneEuroFilter(**self.smoothing_size_config)

    def _add_to_history(self, point):
        self.history.append(point)

    def propagate_prediction(self):
        if self.propagate_count >= self.max_propagate:
            self.reset_history()
            return None

        self.propagate_count += 1
        point = self.history[-1]

        # smooth propagated coordinates
        if point:
            prob, gesture, x, y, side, box, box_size = point
            width, height = box_size
            x, y, width, height = self._apply_filters(x, y, width, height)
            point = (prob, gesture, x, y, side, box, (width, height))

        return point

    def _apply_filters(self, x, y, w, h):
        x, y = int(self.x_filter(x)), int(self.y_filter(y))
        w, h = self.w_filter(w), self.h_filter(h)
        return x, y, w, h

    def _update_prediction(self, prediction):
        prob, gesture, x, y, side, box, box_size = prediction

        width, height = box_size
        x, y, width, height = self._apply_filters(x, y, width, height)

        x_lt = x - width / 2
        y_lt = y - height / 2
        x_rb = x + width / 2
        y_rb = y + height / 2

        box = (x_lt, y_lt, x_rb, y_rb)

        return prob, gesture, x, y, side, box, (width, height)

    def get_current_point(self, new_prediction=None):

        if not new_prediction:
            # if we don't have any detection yet or not enough detection
            if not self.history or len(self.history) < self.min_num_predictions:
                self.reset_history()
                return None

            point = self.propagate_prediction()
            if point:
                self._add_to_history(point)
            return point

        else:
            prediction = self._update_prediction(new_prediction)
            self._add_to_history(prediction)
            self.propagate_count = 0

            if not self.history or len(self.history) < self.min_num_predictions:
                return None

            return prediction

    def reset_history(self):
        self.history = []
        self.propagate_count = 0
