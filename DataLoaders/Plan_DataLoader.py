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

# flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')
# flags.DEFINE_integer('batch_size', 1, 'Batch size. Code currently only handles bs=1')
# flags.DEFINE_string('MIME_dir', '/checkpoint/tanmayshankar/MIME/', 'Data Directory')
flags.DEFINE_enum('arm', 'both', ['left', 'right', 'both'], 'Which arms data to load')

class Plan_Dataset(Dataset):
	'''
	Class implementing instance of dataset class for MIME data. 
	'''
	def __init__(self, opts, split='all'):
		self.opts = opts
		self.split = split
		self.dataset_directory = self.opts.MIME_dir

		# # Must consider permutations of arm and split. 
		# Right Arm: New_Plans / Run*_EE_Plan
		#                      / Run*_Joint_Plan
		#                      / Run*_RG_Traj

		# Left Arm: New_Plans_Left / Run*_EE_Plan
		#                          / Run*_Joint_Plan
		#                          / Run*_LG_traj

		# Both Arms: Ambidextrous_Plans / Run*_EE_Plan
		#                               / Run*_Joint_Plan
		#                               / Run*_Grip_Traj

		# Set these parameters to replace. 
		if self.opts.arm=='left':
			folder = 'New_Plans'
			gripper_suffix = "_LG_Traj"
		elif self.opts.arm=='right':
			folder = 'New_Plans_Left'
			gripper_suffix = "_RG_Traj"
		elif self.opts.arm=='both':
			folder = 'Ambidextrous_Plans'
			gripper_suffix = "_Grip_Traj"

		# Default: /checkpoint/tanmayshankar/MIME/

		if self.split=='all':
			# Collect list of all EE Plans, we will select all Joint Angle Plans correspondingly. 
			self.fulltext = osp.join(self.dataset_directory, 'MIME_jointangles/*/*/New_Plans/Run*_EE_Plan.npy')
			# Joint angle plans filelist is in same order thanks to glob. 
			self.jatext = osp.join(self.dataset_directory, 'MIME_jointangles/*/*/New_Plans/Run*_Joint_Plan.npy')
			# Gripper plans filelist is in same order thanks to glob. 
			# self.rgtext = osp.join(self.dataset_directory, 'MIME_jointangles/*/*/New_Plans/Run*_RG_Traj.npy')

			self.filelist = sorted(glob.glob(self.fulltext))
			self.joint_filelist = sorted(glob.glob(self.jatext))
			# self.gripper_filelist = sorted(glob.glob(self.rgtext))            

		elif self.split=='train':
			self.filelist = np.load(os.path.join(self.dataset_directory,"MIME_jointangles/Plan_Lists/PlanTrainList.npy"))
			self.joint_filelist = np.load(os.path.join(self.dataset_directory,"MIME_jointangles/Plan_Lists/PlanJointTrainList.npy"))
		elif self.split=='val':
			self.filelist = np.load(os.path.join(self.dataset_directory,"MIME_jointangles/Plan_Lists/PlanValList.npy"))
			self.joint_filelist = np.load(os.path.join(self.dataset_directory,"MIME_jointangles/Plan_Lists/PlanJointValList.npy"))
		elif self.split=='test':
			self.filelist = np.load(os.path.join(self.dataset_directory,"MIME_jointangles/Plan_Lists/PlanTestList.npy"))
			self.joint_filelist = np.load(os.path.join(self.dataset_directory,"MIME_jointangles/Plan_Lists/PlanJointTestList.npy"))

		# the loaded np arrays give byte strings, and not strings, which breaks later code
		if not isinstance(self.filelist[0], str):
			self.filelist = [f.decode() for f in self.filelist]
			self.joint_filelist = [f.decode() for f in self.joint_filelist]

		# Now replace terms in filelists based on what arm it is. 
		# The EE file list only needs folder replaced.
		self.filelist = [f.replace("New_Plans",folder).replace('/checkpoint/tanmayshankar/MIME',self.opts.MIME_dir) for f in self.filelist]
		# The Joint file list also only needs folder replaced. 
		self.joint_filelist = [f.replace("New_Plans",folder).replace('/checkpoint/tanmayshankar/MIME',self.opts.MIME_dir) for f in self.joint_filelist]
		# Since we didn't create split lists for Gripper, use the filelist and replace to Gripper. 
		self.gripper_filelist = [f.replace("New_Plans",folder).replace("_EE_Plan",gripper_suffix).replace('/checkpoint/tanmayshankar/MIME',self.opts.MIME_dir) for f in self.filelist]

		# Set joint names.
		self.left_joint_names = ['left_s0','left_s1','left_e0','left_e1','left_w0','left_w1','left_w2']
		self.right_joint_names = ['right_s0','right_s1','right_e0','right_e1','right_w0','right_w1','right_w2']
		self.both_joint_names = self.left_joint_names+self.right_joint_names

	def __len__(self):
		# Return length of file list. 
		return len(self.filelist)

	def __getitem__(self, index):

		file = self.filelist[index]
		joint_file = self.joint_filelist[index]
		gripper_file = self.gripper_filelist[index]

		# Load items.
		elem = {}
		elem['EE_Plan'] = np.load(file)
		elem['JA_Plan'] = np.load(joint_file)
		elem['Grip_Plan'] = np.load(gripper_file)/100

		return elem

# ------------ Data Loader ----------- #
# ------------------------------------ #
def data_loader(opts, split='all', shuffle=True):
	dset = Plan_Dataset(opts, split=split)

	return DataLoader(
		dset,
		batch_size=opts.batch_size,
		shuffle=shuffle,
		num_workers=opts.n_data_workers,
		drop_last=True)

