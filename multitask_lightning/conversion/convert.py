import fire
import torch

from multitask_lightning.conversion import jit_model
from multitask_lightning.conversion import coreml_model

INPUT_SIZE = (1, 3, 128, 256)

def convert(weights_path: str, threshold=0.4):
    jit = jit_model.convert_to_jit(weights_path, threshold, INPUT_SIZE)
    torch.jit.save(jit, "model.pt")

    coreml = coreml_model.convert_to_coreml(jit, INPUT_SIZE)
    coreml.save("model.mlmodel")


if __name__ == "__main__":
    fire.Fire(convert)
