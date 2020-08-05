# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .headers import *
import os.path as osp

flags.DEFINE_integer('n_data_workers', 4, 'Number of data loading workers')
flags.DEFINE_integer('batch_size', 1, 'Batch size. Code currently only handles bs=1')
flags.DEFINE_string('MIME_dir', '/checkpoint/tanmayshankar/MIME/', 'Data Directory')
# flags.DEFINE_boolean('downsampling', True, 'Whether to downsample trajectories. ')
flags.DEFINE_integer('ds_freq', 20, 'Downsample joint trajectories by this fraction. Original recroding rate = 100Hz')
flags.DEFINE_boolean('remote', False, 'Whether operating from a remote server or not.')
# opts = flags.FLAGS

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

class MIME_Dataset(Dataset):
	'''
	Class implementing instance of dataset class for MIME data. 
	'''
	def __init__(self, opts):
		self.dataset_directory = opts.MIME_dir

		# Default: /checkpoint/tanmayshankar/MIME/
		self.fulltext = osp.join(self.dataset_directory, 'MIME_jointangles/*/*/joint_angles.txt')

		if opts.remote:
			self.suff_filelist = np.load(osp.join(self.dataset_directory,"Suffix_Filelist.npy"))
			self.filelist = []
			for j in range(len(self.suff_filelist)):
				self.filelist.append(osp.join(self.dataset_directory,self.suff_filelist[j]))
		else:
			self.filelist = sorted(glob.glob(self.fulltext))

		self.ds_freq = opts.ds_freq

		with open(self.filelist[0], 'r') as file:
			print(self.filelist[0])
			lines = file.readlines()
			self.joint_names = sorted(eval(lines[0].rstrip('\n')).keys())

		self.train_lists = np.load(os.path.join(self.dataset_directory,"MIME_jointangles/Train_Lists.npy"))
		self.val_lists = np.load(os.path.join(self.dataset_directory,"MIME_jointangles/Val_Lists.npy"))
		self.test_lists = np.load(os.path.join(self.dataset_directory,"MIME_jointangles/Test_Lists.npy"))

	def __len__(self):
		# Return length of file list. 
		return len(self.filelist)

	def setup_splits(self):
		self.train_filelist = [] 
		self.val_filelist = [] 
		self.test_filelist = [] 

		for i in range(20):
			self.train_filelist.extend(self.train_lists[i])
			self.val_filelist.extend(self.val_lists[i])
			self.test_filelist.extend(self.test_lists[i])                      

	def getit(self, index, split=None, return_plan_run=None):
		'''
		# Returns Joint Angles as: 
		# List of length Number_Timesteps, with each element of the list a dictionary containing the sequence of joint angles. 
		# Assumes index is within range [0,len(filelist)-1]
		'''

		if split=="train":
			file = self.train_filelist[index]
		elif split=="val":
			file = self.val_filelist[index]
		elif split=="test":
			file = self.test_filelist[index]
		elif split is None: 
			file = self.filelist[index]
			
		left_gripper = np.loadtxt(os.path.join(os.path.split(file)[0],'left_gripper.txt'))
		right_gripper = np.loadtxt(os.path.join(os.path.split(file)[0],'right_gripper.txt'))

		orig_left_traj = np.load(osp.join(osp.split(file)[0], 'Left_EE.npy'))
		orig_right_traj = np.load(osp.join(osp.split(file)[0], 'Right_EE.npy'))        

		joint_angle_trajectory = []

		folder = "New_Plans"
		if return_plan_run is not None:
			ee_plan = np.load(os.path.join(os.path.split(file)[0],"{0}/Run{1}_EE_Plan.npy".format(folder,return_plan_run)))
			ja_plan = np.load(os.path.join(os.path.split(file)[0],"{0}/Run{1}_Joint_Plan.npy".format(folder,return_plan_run)))

		# Open file. 
		with open(file, 'r') as file:
			lines = file.readlines()
			for line in lines:
				dict_element = eval(line.rstrip('\n'))
				if len(dict_element.keys()) == len(self.joint_names):
					# some files have extra lines with gripper keys e.g. MIME_jointangles/4/12405Nov19/joint_angles.txt
					array_element = np.array([dict_element[joint] for joint in self.joint_names])
					joint_angle_trajectory.append(array_element)

		joint_angle_trajectory = np.array(joint_angle_trajectory)

		n_samples = len(orig_left_traj) // self.ds_freq

		elem = {}
		elem['joint_angle_trajectory'] = resample(joint_angle_trajectory, n_samples)
		elem['left_trajectory'] = resample(orig_left_traj, n_samples)
		elem['right_trajectory'] = resample(orig_right_traj, n_samples)
		elem['left_gripper'] = resample(left_gripper, n_samples)
		elem['right_gripper'] = resample(right_gripper, n_samples)
		elem['path_prefix'] = os.path.split(self.filelist[index])[0]
		elem['JA_Plan'] = ja_plan
		elem['EE_Plan'] = ee_plan

		return elem


	def __getitem__(self, index, split=None, return_plan_run=None):
	# def __getitem__(self, inputs):
		'''
		# Returns Joint Angles as: 
		# List of length Number_Timesteps, with each element of the list a dictionary containing the sequence of joint angles. 
		# Assumes index is within range [0,len(filelist)-1]
		'''

		if split=="train":
			file = self.train_filelist[index]
		elif split=="val":
			file = self.val_filelist[index]
		elif split=="test":
			file = self.test_filelist[index]
		elif split is None: 
			file = self.filelist[index]
			
		left_gripper = np.loadtxt(os.path.join(os.path.split(file)[0],'left_gripper.txt'))
		right_gripper = np.loadtxt(os.path.join(os.path.split(file)[0],'right_gripper.txt'))

		orig_left_traj = np.load(osp.join(osp.split(file)[0], 'Left_EE.npy'))
		orig_right_traj = np.load(osp.join(osp.split(file)[0], 'Right_EE.npy'))        

		joint_angle_trajectory = []

		folder = "New_Plans"
		if return_plan_run is not None:
			ee_plan = np.load(os.path.join(os.path.split(file)[0],"{0}/Run{1}_EE_Plan.npy".format(folder,return_plan_run)))
			ja_plan = np.load(os.path.join(os.path.split(file)[0],"{0}/Run{1}_JA_Plan.npy".format(folder,return_plan_run)))

		# Open file. 
		with open(file, 'r') as file:
			lines = file.readlines()
			for line in lines:
				dict_element = eval(line.rstrip('\n'))
				if len(dict_element.keys()) == len(self.joint_names):
					# some files have extra lines with gripper keys e.g. MIME_jointangles/4/12405Nov19/joint_angles.txt
					array_element = np.array([dict_element[joint] for joint in self.joint_names])
					joint_angle_trajectory.append(array_element)

		joint_angle_trajectory = np.array(joint_angle_trajectory)

		n_samples = len(orig_left_traj) // self.ds_freq

		elem = {}
		elem['joint_angle_trajectory'] = resample(joint_angle_trajectory, n_samples)
		elem['left_trajectory'] = resample(orig_left_traj, n_samples)
		elem['right_trajectory'] = resample(orig_right_traj, n_samples)
		elem['left_gripper'] = resample(left_gripper, n_samples)
		elem['right_gripper'] = resample(right_gripper, n_samples)
		elem['path_prefix'] = os.path.split(self.filelist[index])[0]
		elem['JA_Plan'] = ja_plan
		elem['EE_Plan'] = ee_plan

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
def data_loader(opts, shuffle=True):
	dset = MIME_Dataset(opts)

	return DataLoader(
		dset,
		batch_size=opts.batch_size,
		shuffle=shuffle,
		num_workers=opts.n_data_workers,
		drop_last=True)
