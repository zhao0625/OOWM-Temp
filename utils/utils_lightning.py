from typing import Any
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch
from pytorch_lightning import Callback

import wandb

Tensor = TypeVar("torch.tensor")
T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


def to_rgb_from_tensor(x: Tensor):
    return (x * 0.5 + 0.5).clamp(0, 1)


class ImageLogCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        if trainer.logger:
            with torch.no_grad():
                pl_module.eval()
                images = pl_module.sample_images()
                # TODO Note: keep the same key!
                trainer.logger.experiment.log({"Reconstruction": [wandb.Image(images)]}, commit=False)
