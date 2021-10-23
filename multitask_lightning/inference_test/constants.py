from multitask_lightning.inference_test.vis_utils import load_sticker

STICKERS = {
    (0, "right"): load_sticker(
        "./multitask_lightning/inference_test/stickers/thumb_up.png"
    ),
    (0, "left"): load_sticker(
        "./multitask_lightning/inference_test/stickers/thumb_up.png"
    ),
    (1, "left"): load_sticker(
        "./multitask_lightning/inference_test/stickers/thumb_down.png"
    ),
    (1, "right"): load_sticker(
        "./multitask_lightning/inference_test/stickers/thumb_down.png"
    ),
    (2, "right"): load_sticker(
        "./multitask_lightning/inference_test/stickers/hand_up.png"
    ),
    (2, "left"): load_sticker(
        "./multitask_lightning/inference_test/stickers/hand_up.png"
    ),
}
