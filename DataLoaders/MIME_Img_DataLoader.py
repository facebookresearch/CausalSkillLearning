# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .headers import *
import os.path as osp
import pdb
import scipy.misc

flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')
flags.DEFINE_integer('batch_size', 1, 'Batch size. Code currently only handles bs=1')
flags.DEFINE_string('MIME_dir', '/checkpoint/tanmayshankar/MIME/', 'Data Directory')
flags.DEFINE_string('MIME_imgs_dir', '/checkpoint/shubhtuls/data/MIME/', 'Data Directory')
flags.DEFINE_integer('img_h', 64, 'Height')
flags.DEFINE_integer('img_w', 128, 'Width')
flags.DEFINE_integer('ds_freq', 20, 'Downsample joint trajectories by this fraction. Original recroding rate = 100Hz')


def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]


class MIME_Img_Dataset(Dataset):
	'''
	Class implementing instance of dataset class for MIME data. 
	'''
	def __init__(self, opts, split='all'):
		self.dataset_directory = opts.MIME_dir
		self.imgs_dataset_directory = opts.MIME_imgs_dir
		self.img_h = opts.img_h
		self.img_w = opts.img_w

		# Default: /checkpoint/tanmayshankar/MIME/
		self.fulltext = osp.join(self.dataset_directory, 'MIME_jointangles/*/*/joint_angles.txt')
		self.filelist = glob.glob(self.fulltext)

		self.ds_freq = opts.ds_freq

		with open(self.filelist[0], 'r') as file:
			lines = file.readlines()
			self.joint_names = sorted(eval(lines[0].rstrip('\n')).keys())

		if split == 'all':
			self.filelist = self.filelist
		else:
			self.task_lists = np.load(os.path.join(
				self.dataset_directory, 'MIME_jointangles/{}_Lists.npy'.format(split.capitalize())))
			self.filelist = []
			for i in range(20):
				self.filelist.extend(self.task_lists[i])
			self.filelist = [f.replace('/checkpoint/tanmayshankar/MIME/', opts.MIME_dir) for f in self.filelist]

	def __len__(self):
		# Return length of file list. 
		return len(self.filelist)

	def __getitem__(self, index):
		'''
		# Returns Joint Angles as: 
		# List of length Number_Timesteps, with each element of the list a dictionary containing the sequence of joint angles. 
		# Assumes index is within range [0,len(filelist)-1]
		'''
		file = self.filelist[index]
		file_split = file.split('/')
		frames_folder = osp.join(self.imgs_dataset_directory, file_split[-3], file_split[-2], 'frames')
		n_frames = len(os.listdir(frames_folder))

		imgs = []
		frame_inds = [0, n_frames//2, n_frames-1]
		for fi in frame_inds:
			img = scipy.misc.imread(osp.join(frames_folder, 'im_{}.png'.format(fi+1)))
			imgs.append(scipy.misc.imresize(img, (self.img_h, self.img_w)))
		imgs = np.stack(imgs)

		left_gripper = np.loadtxt(os.path.join(os.path.split(file)[0],'left_gripper.txt'))
		right_gripper = np.loadtxt(os.path.join(os.path.split(file)[0],'right_gripper.txt'))

		joint_angle_trajectory = []
		# Open file. 
		with open(file, 'r') as file:
			lines = file.readlines()
			for line in lines:
				dict_element = eval(line.rstrip('\n'))
				if len(dict_element.keys()) == len(self.joint_names):
					array_element = np.array([dict_element[joint] for joint in self.joint_names])
					joint_angle_trajectory.append(array_element)

		joint_angle_trajectory = np.array(joint_angle_trajectory)

		n_samples = len(joint_angle_trajectory) // self.ds_freq

		elem = {}
		elem['imgs'] = imgs
		elem['joint_angle_trajectory'] = resample(joint_angle_trajectory, n_samples)
		elem['left_gripper'] = resample(left_gripper, n_samples)/100
		elem['right_gripper'] = resample(right_gripper, n_samples)/100
		elem['is_valid'] = int(np.linalg.norm(np.diff(elem['joint_angle_trajectory'],axis=0),axis=1).max() < 1.0)

		return elem

	def recreate_dictionary(self, arm, joint_angles):
		if arm=="left":
			offset = 2
			width = 7 
		elif arm=="right":
			offset = 9
			width = 7
		elif arm=="full":
			offset = 0
			width = len(self.joint_names)
		return dict((self.joint_names[i],joint_angles[i-offset]) for i in range(offset,offset+width))

# ------------ Data Loader ----------- #
# ------------------------------------ #

def data_loader(opts, split='all', shuffle=True):
	dset = MIME_Img_Dataset(opts, split=split)

	return DataLoader(
		dset,
		batch_size=opts.batch_size,
		shuffle=shuffle,
		num_workers=opts.n_data_workers,
		drop_last=True)
