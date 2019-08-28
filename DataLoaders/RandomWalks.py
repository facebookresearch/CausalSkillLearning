from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import random as stdlib_random, string

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import numpy as np

from absl import flags, app

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from ..utils import plotting as plot_util

flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')
flags.DEFINE_integer('batch_size', 1, 'Batch size. Code currently only handles bs=1')
flags.DEFINE_integer('n_segments_min', 4, 'Min Number of gt segments per trajectory')
flags.DEFINE_integer('n_segments_max', 4, 'Max number of gt segments per trajectory')

dirs_2d = np.array([
    [1,0],
    [0,1],
    [-1,0],
    [0,-1]
])


def vis_walk(walk):
    '''
    Args:
        walk: (nT+1) X 2 array
    Returns:
        im: 200 X 200 X 4 numpy array
    '''

    t = walk.shape[0]
    xs = walk[:,0]
    ys = walk[:,1]
    color_inds = np.linspace(0, 255, t).astype(np.int).tolist()
    cs = plot_util.colormap[color_inds, :]

    fig = plt.figure(figsize=(4, 4), dpi=50)
    ax = fig.subplots()

    ax.scatter(xs, ys, c=cs)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal', 'box')

    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False)

    ax.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False)

    fig.tight_layout()
    fname = '/tmp/' + ''.join(stdlib_random.choices(string.ascii_letters, k=8)) + '.png'
    fig.savefig(fname)
    plt.close(fig)

    im = plt.imread(fname)
    os.remove(fname)

    return im


def walk_segment(origin, direction, n_steps=10, step_size=0.1, noise=0.02, rng=None):
    '''
    Args:
        origin: nd numpy array
        direction: nd numpy array with unit norm
        n_steps: length of time seq
        step_size: size of each step
        noise: magintude of max actuation noise
    Returns:
        segment: n_steps X nd array
            note that the first position in segment is different from origin
    '''
    if rng is None:
        rng = np.random

    nd = origin.shape[0]
    segment = np.zeros((n_steps, nd)) + origin
    segment += np.arange(1, n_steps+1).reshape((-1,1))*direction*step_size
    segment += rng.uniform(low=-1, high=1, size=(n_steps, nd)) * noise/nd
    return segment


def random_walk2d(origin, num_segments=4, rng=None):
    '''
    Args:
        origin: 2d numpy array
        num_segments: length of time seq
    Returns:
        walk: (nT+1) X 2 array
    '''
    if rng is None:
        rng = np.random

    dir_ind = rng.randint(4)
    walk = origin.reshape(1,2)
    seg_lengths = []
    for s in range(num_segments):
        seg_length = rng.randint(6,10)
        seg_lengths.append(seg_length)
        step_size = 0.1 + (rng.uniform() - 0.5)*0.05

        segment = walk_segment(origin, dirs_2d[dir_ind], n_steps=seg_length, step_size=step_size, rng=rng)
        origin = segment[-1]
        walk = np.concatenate((walk, segment), axis=0)

        dir_ind += 2 * rng.randint(2) -1
        dir_ind = dir_ind % 4

    return walk, seg_lengths


class RandomWalksDataset(Dataset):

    def __init__(self, opts):
        self.opts = opts
        self.n_segments_min = self.opts.n_segments_min
        self.n_segments_max = self.opts.n_segments_max

    def __len__(self):
        return int(1e6)

    def __getitem__(self, ix):
        rng = np.random.RandomState(ix)
        ns = rng.randint(self.n_segments_min, self.n_segments_max+1)
        trajectory, self.seg_lengths_ix = random_walk2d(np.zeros(2), num_segments=ns, rng=rng)
        return trajectory

# ------------ Data Loader ----------- #
# ------------------------------------ #
def data_loader(opts, shuffle=True):
    dset = RandomWalksDataset(opts)

    return DataLoader(
        dset,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        num_workers=opts.n_data_workers,
        drop_last=True)


if __name__ == '__main__':
    walk = random_walk2d(np.zeros(2), num_segments=4)
    print(walk)
