import copy
import os

import torch
import yaml
from omegaconf import DictConfig

import algorithms.deprecated_world_model
from algorithms import world_model
from scripts.init import ex


# TODO deprecated
# @ex.capture
# def load_pretrained_model(_log, _run, model_train, pretrain_save_folder, input_shape, cuda):
#     # [if load saved model and config exists]
#     assert os.path.isfile(os.path.join(pretrain_save_folder, 'model_config.yaml'))
#
#     with open(os.path.join(pretrain_save_folder, 'model_config.yaml'), 'r') as fp:
#         model_train_loaded = yaml.load(fp, Loader=yaml.FullLoader)
#         _log.warning('Using loaded config:')
#         _log.warning(model_train_loaded)  # TODO
#         print()
#
#     # [convert]
#     model_train_loaded = DictConfig(model_train_loaded)
#     model_train_loaded.input_dims = list(input_shape)
#
#     # TODO [much have N filters and N actions, or K filters and K actions]
#     assert model_train_loaded.action_mapping == model_train_loaded.extra_filter
#
#     device = torch.device('cuda' if cuda else 'cpu')
#
#     # TODO only for slot-att version
#     assert model_train_loaded.homo_slot_att
#
#     diff = set(model_train_loaded.items()).symmetric_difference(model_train_loaded.items())
#     _log.warning(f'Different values: {diff}\n')
#     _log.warning(f"Number of total objects for these models: (load=) {model_train_loaded['num_objects_total']} =?= "
#                  f"(train=) {model_train['num_objects_total']}\n")
#
#     model_for_load = algorithms.deprecated_world_model.PixelContrastiveSWM(**model_train_loaded).to(device)
#     # # TODO load - mainly load the encoder (and decoder)
#     model_file = os.path.join(pretrain_save_folder, 'model.pt')
#     model_for_load.load_state_dict(torch.load(model_file))
#
#     # TODO load from command line input
#     model = algorithms.deprecated_world_model.PixelContrastiveSWM(**model_train).to(device)
#
#     # FIXME test just use encoder?
#
#     model.encoder = copy.deepcopy(model_for_load.encoder)
#     del model_for_load
#     # FIXME memory
#     # FIXME need to work on object_identity? or maybe just generate during
#
#     # FIXME froze weight?
#     # for _param in model.encoder.parameters():
#     #     _param.requires_grad = False
#     #
#     # # unfroze these
#     # modules = [
#     #     model.encoder.net.action_encoder,
#     #     model.encoder.net.identity_encoder,
#     #     model.encoder.net.slot_attention.project_k_action,
#     #     model.encoder.net.slot_attention.project_q_action,
#     #     model.encoder.net.slot_attention.project_v_action
#     # ]
#     # for _module in modules:
#     #     for _param in model.parameters():
#     #         _param.requires_grad = True
#
#     # FIXME
#     model.encoder.update_after_load(model_train['num_objects_total'])
#
#     print("model_train_loaded['num_objects_total'], model_train['num_objects_total']",
#           model_train_loaded['num_objects_total'], model_train['num_objects_total'])
#     print("model.encoder.net.num_objects_total, model.encoder.net.object_identity",
#           model.encoder.net.num_objects_total, model.encoder.net.object_identity.size())
#
#     # TODO use input config but load weights
#     # model = modules_cswm.PixelContrastiveSWM(**model_train).to(device)
#     #
#     # model_file = os.path.join(pretrain_save_folder, 'model.pt')
#     # model_weight = torch.load(model_file)
#     #
#     # model_weight_load = {}
#     # for key in model_weight.keys():
#     #     if key.startwith('encoder'):
#     #         model_weight_load[key] = model_weight[key]
#     #
#     # model.load_state_dict(model_weight_load)
#
#     model.eval()
#
#     return model
