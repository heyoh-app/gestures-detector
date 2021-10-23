import albumentations as albu


class Transforms:

    def __init__(self, size, scope='weak', size_transform='center', border_mode=0):
        self.scope = scope
        self.size_transform = size_transform
        self.border_mode = border_mode
        self.size = size

    def define_augs(self):
        augs = {
            'strong': albu.Compose([
                albu.ShiftScaleRotate(
                    rotate_limit=10,
                    shift_limit=0.05,
                    scale_limit=(0., 0.1),
                    border_mode=self.border_mode,
                    p=0.5
                ),
                albu.OneOf([
                    albu.CLAHE(clip_limit=2),
                    albu.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                    albu.RandomGamma(),
                ], p=0.5),
                albu.OneOf([
                    albu.MotionBlur(p=0.2),
                    albu.MedianBlur(blur_limit=3, p=0.1),
                    albu.Blur(blur_limit=3, p=0.1),
                ], p=0.4),
                albu.OneOf(
                    [
                        albu.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=.4),
                        albu.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, val_shift_limit=30, p=.4)
                    ],
                    p=.2
                ),
                albu.OneOf([
                    albu.GaussNoise(),
                    albu.ISONoise(intensity=(0.1, 1.0), p=0.6)
                ], p=0.5),
                albu.RandomShadow(p=0.2)
            ]),
            'weak': albu.Compose([albu.core.transforms_interface.NoOp(),
                                  ]),
        }

        size_transforms = {
            'train': albu.Compose([
                albu.SmallestMaxSize(max_size=self.size, always_apply=True),
                albu.RandomCrop(height=self.size, width=self.size)
            ], p=1),
            'val': albu.Compose([
                albu.LongestMaxSize(max_size=self.size, always_apply=True),
                albu.PadIfNeeded(min_height=self.size, min_width=self.size)
            ], p=1),
        }
        return augs, size_transforms

    def get_augmented(self, data):
        augs, size_transforms = self.define_augs()
        aug_fn = augs[self.scope]
        size_fn = size_transforms[self.size_transform]

        normalize = albu.Normalize(mean=[0.449, 0.449, 0.449], std=[0.226, 0.226, 0.226])

        pipeline = albu.Compose(
            [size_fn, aug_fn, normalize],
            p=1,
            bbox_params=albu.BboxParams(format='pascal_voc', min_visibility=0.5),
        )

        augmented = pipeline(**data)
        image = augmented["image"]
        bboxes = augmented["bboxes"]

        return image, bboxes
