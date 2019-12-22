#!/usr/bin/env python
import numpy as np
import glob, os, sys, argparse
import torch, copy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from IPython import embed

import matplotlib
matplotlib.use('Agg')
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
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.animation import FuncAnimation
import tensorflow as tf
import tempfile
import moviepy.editor as mpy
import subprocess
import h5py