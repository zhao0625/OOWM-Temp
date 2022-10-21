import copy
import datetime
import os
from collections import defaultdict

import torch
import wandb
import yaml
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from torch.utils import data
import numpy as np

from scripts.init import ex
import utils.utils_cswm as utils

from scripts.helpers_model import get_model

from algorithms.trainer_wrapper import ShapesDataModule, SlotAttentionNetworkModule


@ex.capture
def init_data_model_trainer(_log, model_train, model_eval):
    model_train = DictConfig(model_train)

    # > Init model
    model_object = get_model(load=False)
    model_object.apply(utils.weights_init)  # > Init model after creating

    # > Init data
    data_module = ShapesDataModule(
        train_batch_size=model_train.representation.batch_size,
        val_batch_size=model_train.representation.batch_size,
        num_workers=model_train.num_workers,
        train_data_path=model_train['dataset'],
        val_data_path=model_eval['dataset'],
    )

    # > Init trainer module
    network_module = SlotAttentionNetworkModule(
        model=model_object.slot_attention,  # > only need to train the Slot Attention part
        datamodule=data_module,
        config=model_train.representation  # > config from representation config
    )

    return network_module, data_module
