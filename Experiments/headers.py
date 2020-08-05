# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import glob, os, sys, argparse
import torch, copy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from IPython import embed

import matplotlib
matplotlib.use('Agg')
# matplotlib.rcParams['animation.ffmpeg_args'] = '-report'
matplotlib.rcParams['animation.bitrate'] = 2000
import matplotlib.pyplot as plt
import tensorboardX
from scipy import stats
from absl import flags
from memory_profiler import profile
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from IPython import embed
import pdb
import sklearn.manifold as skl_manifold
from sklearn.decomposition import PCA
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.animation import FuncAnimation
import tensorflow as tf
import tempfile
import moviepy.editor as mpy
import subprocess
import h5py
import time
import robosuite
import unittest
import cProfile

from scipy import stats, signal
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks, argrelextrema

from sklearn.neighbors import NearestNeighbors
