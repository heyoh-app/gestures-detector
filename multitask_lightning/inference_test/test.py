import cv2
import fire
import time
import torch
import numpy as np

import pyglview
import acapture

from multitask_lightning.inference_test.utils import predict
from multitask_lightning.inference_test.utils import Track
from multitask_lightning.inference_test.vis_utils import (
    overlay_transparent,
    image_resize,
)
from multitask_lightning.inference_test.constants import STICKERS

import coremltools as ct

torch.set_grad_enabled(False)


def load_model(model_path: str, use_jit: bool):
    if not use_jit:
        model = ct.models.MLModel(model_path)
        return model

    model = torch.jit.load(model_path, map_location="cpu")
    return model


def run(model_path: str, camera_id=2, debug: bool = True, smooth: bool = True):
    use_jit = "mlmodel" not in model_path
    model = load_model(model_path, use_jit)
    if debug:
        capture_cv2(model, camera_id, use_jit, smooth)
    else:
        capture_acapture(model, camera_id, use_jit, smooth)


def capture_acapture(
    model: torch.jit.TracedModule, camera_id: int, use_jit: bool, smooth: bool
):
    viewer = pyglview.Viewer()
    cap = acapture.open(camera_id)

    right_hand_track = Track(hand_side="right")
    left_hand_track = Track(hand_side="left")

    def loop():
        check, frame = cap.read()
        start_time = time.time()
        predictions = predict(frame, model, use_jit)
        predictions = update_predictions(
            predictions, right_hand_track, left_hand_track, smooth
        )
        FPS = 1 / (time.time() - start_time)
        frame = process_frame(frame, predictions, FPS, False)
        if check:
            viewer.set_image(frame)

    viewer.set_loop(loop)
    viewer.start()


def capture_cv2(
    model: torch.jit.TracedModule, camera_id: int, use_jit: bool, smooth: bool
):
    vid = cv2.VideoCapture(camera_id)
    right_hand_track = Track(hand_side="right")
    left_hand_track = Track(hand_side="left")

    while True:
        ret, frame = vid.read()
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        start_time = time.time()
        predictions = predict(img, model, use_jit)
        predictions = update_predictions(
            predictions, right_hand_track, left_hand_track, smooth
        )
        FPS = 1 / (time.time() - start_time)

        frame = process_frame(frame, predictions, FPS, True)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def update_predictions(predictions, right_hand_track, left_hand_track, smooth):
    if not smooth:
        return predictions

    final_predictions = []
    right_hand_found = False
    left_hand_found = False
    for prediction in predictions:
        _, gesture, _, _, side, _, _ = prediction

        # faces
        if gesture < 2:
            final_predictions.append(prediction)

        elif 2 <= gesture <= 5:
            side = "right" if side > 0.5 else "left"
            if side == "right":
                prediction = right_hand_track.get_current_point(prediction)
                right_hand_found = True
            else:
                prediction = left_hand_track.get_current_point(prediction)
                left_hand_found = True
            final_predictions.append(prediction)

    if right_hand_found == False:
        prediction = right_hand_track.get_current_point()
        final_predictions.append(prediction)

    if left_hand_found == False:
        prediction = left_hand_track.get_current_point()
        final_predictions.append(prediction)

    # filter None
    final_predictions = [i for i in final_predictions if i]
    return final_predictions


def process_frame(frame, predictions, FPS, debug) -> np.ndarray:
    for hand in predictions:
        _, gesture, center_x, center_y, side, box, box_size = hand
        side = "right" if side > 0.5 else "left"
        if (gesture - 2, side) in STICKERS:
            sticker = STICKERS[(gesture - 2, side)]
            sticker = image_resize(
                sticker, height=int(2.5 * max(box_size[0], box_size[1]))
            )
            sticker = sticker if side == "left" else cv2.flip(sticker, 1)

            xtl = center_x - (sticker.shape[0] // 2)
            ytl = center_y - (sticker.shape[1] // 2)
            xtl = xtl if xtl > 0 else 0
            ytl = ytl if ytl > 0 else 0
            _ = overlay_transparent(frame, sticker, xtl, ytl)
        else:
            colors = {
                0: (255, 255, 255),
                1: (0, 255, 0),
            }
            cv2.circle(frame, tuple([center_x, center_y]), 6, colors[gesture], 4)

        if debug:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (100, 100)
    fontScale = 1
    fontColor = (0, 255, 255)
    lineType = 2
    frame = cv2.flip(frame, 1)

    cv2.putText(
        frame,
        "{}FPS".format(int(FPS)),
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType,
    )

    return frame


if __name__ == "__main__":
    fire.Fire(run)
