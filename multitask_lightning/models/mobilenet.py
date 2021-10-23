import torchvision
import torch.nn as nn
from segmentation_models_pytorch.encoders._base import EncoderMixin

class MobileNetV2ClippedEncoder(torchvision.models.MobileNetV2, EncoderMixin):

    def __init__(self, out_channels, depth=4, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3
        del self.classifier
        del self.features[11:]

    def get_stages(self):
        return [
            nn.Identity(),
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:11],
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("classifier.1.bias")
        state_dict.pop("classifier.1.weight")
        super().load_state_dict(state_dict, strict=False, **kwargs)


mobilenet_clipped_encoders = {
    "mobilenet_v2_clipped": {
        "encoder": MobileNetV2ClippedEncoder,
        "pretrained_settings": {
            "imagenet": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                "input_space": "RGB",
                "input_range": [0, 1],
            },
        },
        "params": {
            "out_channels": (3, 16, 24, 32, 64),
        },
    },
}