import torch
from torch import nn

import segmentation_models_pytorch as smp
from multitask_lightning.models.model import UnetClipped


class PredictionPipeline(nn.Module):
    def __init__(self, threshold: float):
        super(PredictionPipeline, self).__init__()
        self.threshold = nn.Threshold(threshold, 0)
        self.side_activation = torch.nn.Sigmoid()

        self.net = _load_model()

        self.kernel = 3
        self.output_stride = 2
        self.scale = 5
        self.pad = (self.kernel - 1) // 2
        self.maxpool = nn.MaxPool2d(
            (self.kernel, self.kernel), stride=1, padding=self.pad
        )

    def forward(self, x):
        feature = self.net(x)
        heatmap, side, sizes = torch.split(
            feature, split_size_or_sections=[5, 1, 1], dim=1
        )

        activated = torch.sigmoid(heatmap)
        thresholded = self.threshold(activated)

        # nms
        pooled = self.maxpool(thresholded)
        mask = torch.eq(pooled, thresholded).float()
        masked = torch.mul(thresholded, mask).squeeze()

        side_activated = self.side_activation(side).squeeze()

        # hotfix: explicitly filter values > 0, torch.mul produces extremely small values like -9.766221e-05
        masked = masked > 0
        coords = masked.nonzero()
        sizes_activated = torch.sigmoid(sizes).squeeze()

        binary_mask = masked > 0

        probs = torch.max(activated.squeeze(), dim=0).values
        probs = probs.masked_select(binary_mask)

        size_width = sizes_activated.masked_select(binary_mask)
        size_height = sizes_activated.masked_select(binary_mask)

        side_squeezed = side_activated.masked_select(binary_mask)

        # Dumb heuristics to keep the output names
        probs.squeeze()
        coords.squeeze()
        size_width.squeeze()
        side_squeezed.squeeze()
        size_height.squeeze()

        return probs, coords, size_width, size_height, side_squeezed


def _load_model():
    train_model = UnetClipped(
        "mobilenet_v2_clipped",
        encoder_depth=4,
        decoder_depth=3,
        decoder_channels=[128, 64, 32],
        classes=7,
        encoder_weights="imagenet",
        in_channels=3,
    )
    train_model.eval()
    return train_model


def convert_to_jit(weights_path: str, threshold: float, input_size: tuple) -> torch.jit.ScriptModule:
    model = PredictionPipeline(threshold=threshold)
    model_dict = torch.load(weights_path, map_location="cpu")["state_dict"]
    model.load_state_dict(model_dict, strict=False)
    example_input = torch.rand(input_size)
    model = torch.jit.trace(model, example_inputs=example_input)
    return model
