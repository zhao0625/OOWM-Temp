"""Utility functions."""

import os
from collections import namedtuple

import h5py
import numpy as np

import torch
from torch.utils import data
from torch import nn

import matplotlib.pyplot as plt

EPS = 1e-17

import argparse


def css_to_ssc(image):
    return image.transpose((1, 2, 0))


def to_np(x):
    return x.detach().cpu().numpy()


def save_image_box_world(img, fname, t, action):
    import os
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig2 = plt.imshow(img / 255, interpolation='none')
    fig2.axes.get_xaxis().set_visible(False)
    fig2.axes.get_yaxis().set_visible(False)
    plt.savefig(
        os.path.join(fname, 'observation_{}_{}.png'.format(t, action)),
        dpi=20, bbox_inches='tight', pad_inches=0.1
    )


def save_dict(dict_data: dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    import os, h5py
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as f:
        for key, data in dict_data.items():
            f.create_dataset(key, data=data)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def save_dict_h5py(array_dict, fname):
    """Save dictionary containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for key in array_dict.keys():
            hf.create_dataset(key, data=array_dict[key])


def load_dict_h5py(fname):
    """Restore dictionary containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for key in hf.keys():
            array_dict[key] = hf[key][:]
    return array_dict


def save_list_dict_h5py(array_dict, fname):
    """Save list of dictionaries containing numpy arrays to h5py file."""

    # Ensure directory exists
    directory = os.path.dirname(fname)

    print('>>> directory:', directory, os.path.abspath(directory))

    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(fname, 'w') as hf:
        for i in array_dict.keys():
            grp = hf.create_group(str(i))
            for key in array_dict[i].keys():
                grp.create_dataset(key, data=array_dict[i][key])


def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = dict()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            # > Handle other string keys, such as for visualization
            idx = i if grp.isdecimal() else grp

            array_dict[idx] = dict()
            for key in hf[grp].keys():
                array_dict[idx][key] = hf[grp][key][:]

    return array_dict


def get_colors(cmap='Set1', num_colors=9):
    """Get color array from matplotlib colormap."""
    cm = plt.get_cmap(cmap)

    colors = []
    for i in range(num_colors):
        colors.append((cm(1. * i / num_colors)))

    return colors


def pairwise_distance_matrix(x, y, verbose=True):
    if verbose:
        print('> Sizes:', x.size(), y.size(), x.dtype, y.dtype)

    num_samples = x.size(0)
    dim = x.size(1)

    x = x.unsqueeze(1).expand(num_samples, num_samples, dim)
    y = y.unsqueeze(0).expand(num_samples, num_samples, dim)

    return torch.pow(x - y, 2).sum(2)


def get_act_fn(act_fn):
    if act_fn == 'relu':
        return nn.ReLU()
    elif act_fn == 'leaky_relu':
        return nn.LeakyReLU()
    elif act_fn == 'elu':
        return nn.ELU()
    elif act_fn == 'sigmoid':
        return nn.Sigmoid()
    elif act_fn == 'softplus':
        return nn.Softplus()
    else:
        raise ValueError('Invalid argument for `act_fn`.')


def to_one_hot(indices, max_index):
    """Get one-hot encoding of index tensors."""
    zeros = torch.zeros(
        indices.size()[0], max_index, dtype=torch.float32,
        device=indices.device)
    return zeros.scatter_(1, indices.unsqueeze(1), 1)


def to_float(np_array):
    """Convert numpy array to float32."""
    return np.array(np_array, dtype=np.float32)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


class StateTransitionsDataset(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file, action_mapping=True):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        self.action_mapping = action_mapping  # > deprecated, was used for different action

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])
        action = self.experience_buffer[ep]['action' if self.action_mapping else 'action_unmap'][step]

        return obs, action, next_obs


TransitionWithNeg = namedtuple('TransitionWithNeg', ['obs', 'action', 'next_obs', 'neg_obs'])


class StateTransitionsDatasetObjConfig(data.Dataset):
    """Create dataset of (o_t, a_t, o_{t+1}) transitions from replay buffer."""

    def __init__(self, hdf5_file, same_config_ratio=0.5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience buffer
        """

        # > Note: now return dict
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        print('>>> Finish loading into memory')

        # > include obj visualization
        if 'obj_vis' in self.experience_buffer:
            print('> Visualization keys:', self.experience_buffer['obj_vis'].keys())
            self.obj_vis = self.experience_buffer['obj_vis']['column']
            del self.experience_buffer['obj_vis']

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

        self.same_config_ratio = same_config_ratio

        # > save two lists of episodes, with same or different configurations (scenes/ids) with that scene
        # > O(N^2) complexity - could be optimized
        self.same_config_ep_list_all = []
        self.diff_config_list_all = []
        for ep_all in range(len(self.experience_buffer)):
            same_config_ep_list = []
            diff_config_list = []
            for ep in range(len(self.experience_buffer)):
                if tuple(np.sort(self.experience_buffer[ep_all]['ids'][0])) == tuple(
                        np.sort(self.experience_buffer[ep]['ids'][0])):
                    same_config_ep_list.append(ep)
                else:
                    diff_config_list.append(ep)
            self.same_config_ep_list_all.append(same_config_ep_list)
            self.diff_config_list_all.append(diff_config_list)

        self.episode_len = self.experience_buffer[0]['obs'].shape[0]

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        ep, step = self.idx2episode[idx]

        obs = to_float(self.experience_buffer[ep]['obs'][step])
        action = self.experience_buffer[ep]['action'][step]
        next_obs = to_float(self.experience_buffer[ep]['next_obs'][step])

        # > For N=K, only should have one possible config, so use the same config
        if len(self.diff_config_list_all[ep]) == 0:
            same_config_ep = self.same_config_ep_list_all[ep]
            neg_ep = np.random.choice(same_config_ep)

        # > with some probability, use same config in negative sampling
        elif np.random.rand() < self.same_config_ratio:
            same_config_ep = self.same_config_ep_list_all[ep]
            neg_ep = np.random.choice(same_config_ep)

        # > otherwise, some probability to use different config
        else:
            diff_config_ep = self.diff_config_list_all[ep]
            neg_ep = np.random.choice(diff_config_ep)

        # sample a random step - avoid same image
        rand_step = np.random.randint(self.episode_len)
        neg_obs = to_float(self.experience_buffer[neg_ep]['obs'][rand_step])

        return TransitionWithNeg(obs, action, next_obs, neg_obs)


class PathDataset(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, action_mapping, path_length=5):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
                buffer
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        self.path_length = path_length
        self.action_mapping = action_mapping  # > deprecated

        # > include obj visualization
        if 'obj_vis' in self.experience_buffer:
            print('> Visualization keys:', self.experience_buffer['obj_vis'].keys())
            self.obj_vis = self.experience_buffer['obj_vis']['column']
            del self.experience_buffer['obj_vis']

    def __len__(self):
        return len(self.experience_buffer)

    def __getitem__(self, idx):
        observations = []
        actions = []
        for i in range(self.path_length):
            obs = to_float(self.experience_buffer[idx]['obs'][i])
            action = self.experience_buffer[idx]['action' if self.action_mapping else 'action_unmap'][i]
            observations.append(obs)
            actions.append(action)
        obs = to_float(
            self.experience_buffer[idx]['next_obs'][self.path_length - 1])
        observations.append(obs)
        return observations, actions


class SegmentedPathDataset(data.Dataset):
    """Create dataset of {(o_t, a_t)}_{t=1:N} paths from replay buffer.
    """

    def __init__(self, hdf5_file, action_mapping, segment_length=None):
        """
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience buffer
            segment_length: if not None, use this number to segment the length, used for evaluation on training data
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)
        # self.path_length = path_length
        self.action_mapping = action_mapping

        self.segment_length = segment_length
        if segment_length is not None:
            self.episode_length = len(self.experience_buffer[0]['action'])
            # assert len(self.experience_buffer[0]['obs']) == self.episode_length + 1

            self.num_segments = self.episode_length // self.segment_length
            print('> Segmented dataset! num_segments per episode:', self.num_segments)

        # > include obj visualization
        if 'obj_vis' in self.experience_buffer:
            print('> Visualization keys:', self.experience_buffer['obj_vis'].keys())
            self.obj_vis = self.experience_buffer['obj_vis']['column']
            del self.experience_buffer['obj_vis']

    def id2segment(self, idx, step):
        ep_id = idx // self.num_segments
        step_shift = self.num_segments * (idx % self.num_segments)
        step = step + step_shift
        return ep_id, step

    def __len__(self):
        if self.segment_length is None:
            return len(self.experience_buffer)
        else:
            return len(self.experience_buffer) * self.segment_length

    def __getitem__(self, idx):
        observations = []
        actions = []

        # > Index to the the corresponding segment
        for i in range(self.segment_length):
            ep_id, step = self.id2segment(idx=idx, step=i)

            obs = to_float(self.experience_buffer[ep_id]['obs'][step])
            action = self.experience_buffer[ep_id]['action' if self.action_mapping else 'action_unmap'][step]

            observations.append(obs)
            actions.append(action)

        ep_id, step = self.id2segment(idx=idx, step=self.segment_length - 1)
        obs = to_float(self.experience_buffer[ep_id]['next_obs'][step])

        observations.append(obs)

        return observations, actions


class ObsOnlyDataset(data.Dataset):
    def __init__(self, hdf5_file):
        """
        A dataset only provide observation images
        Args:
            hdf5_file (string): Path to the hdf5 file that contains experience
        """
        self.experience_buffer = load_list_dict_h5py(hdf5_file)

        if 'obj_vis' in self.experience_buffer:
            del self.experience_buffer['obj_vis']

        # Build table for conversion between linear idx -> episode/step idx
        self.idx2episode = list()
        step = 0
        for ep in range(len(self.experience_buffer)):
            num_steps = len(self.experience_buffer[ep]['action'])
            idx_tuple = [(ep, idx) for idx in range(num_steps)]
            self.idx2episode.extend(idx_tuple)
            step += num_steps

        self.num_steps = step

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        """
        Only return obs as training data
        """
        ep, step = self.idx2episode[idx]
        obs = to_float(self.experience_buffer[ep]['obs'][step])
        return obs


def args_ini():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                        help='Learning rate.')

    parser.add_argument('--encoder', type=str, default='small',
                        help='Object extrator CNN size (e.g., `small`).')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Energy scale.')
    parser.add_argument('--hinge', type=float, default=1.,
                        help='Hinge threshold parameter.')

    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='Number of hidden units in transition MLP.')
    parser.add_argument('--embedding-dim', type=int, default=2,
                        help='Dimensionality of embedding.')
    parser.add_argument('--action-dim', type=int, default=4,
                        help='Dimensionality of action space.')
    parser.add_argument('--num-objects', type=int, default=5,
                        help='Number of object slots in model.')
    parser.add_argument('--ignore-action', action='store_true', default=False,
                        help='Ignore action in GNN transition model.')
    parser.add_argument('--copy-action', action='store_true', default=False,
                        help='Apply same action to all object slots.')

    parser.add_argument('--decoder', action='store_true', default=False,
                        help='Train model using decoder and pixel-based loss.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disable CUDA training.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42).')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='How many batches to wait before logging'
                             'training status.')
    parser.add_argument('--dataset', type=str,
                        default='data/shapes_train.h5',
                        help='Path to replay buffer.')
    parser.add_argument('--name', type=str, default='none',
                        help='Experiment name.')
    parser.add_argument('--save-folder', type=str,
                        default='checkpoints',
                        help='Path to checkpoints.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
