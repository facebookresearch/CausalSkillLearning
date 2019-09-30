#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *
import os.path as osp

def select_baxter_angles(trajectory, joint_names, arm='right'):
    # joint names in order as used via mujoco visualizer
    baxter_joint_names = ['right_s0', 'right_s1', 'right_e0', 'right_e1', 'right_w0', 'right_w1', 'right_w2', 'left_s0', 'left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
    if arm == 'right':
        select_joints = baxter_joint_names[:7]
    elif arm == 'left':
        select_joints = baxter_joint_names[7:]
    elif arm == 'both':
        select_joints = baxter_joint_names
    inds = [joint_names.index(j) for j in select_joints]
    return trajectory[:, inds]


def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]


class MIME_Dataset(Dataset):
	'''
	Class implementing instance of dataset class for MIME data. 
	'''
	def __init__(self, split='all'):
		self.dataset_directory = '/checkpoint/tanmayshankar/MIME/'
		self.ds_freq = 20

		# Default: /checkpoint/tanmayshankar/MIME/
		self.fulltext = osp.join(self.dataset_directory, 'MIME_jointangles/*/*/joint_angles.txt')
		self.filelist = glob.glob(self.fulltext)

		

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
			self.filelist = [f.replace('/checkpoint/tanmayshankar/MIME/', self.dataset_directory) for f in self.filelist]
		# print(len(self.filelist))

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

		left_gripper = np.loadtxt(os.path.join(os.path.split(file)[0],'left_gripper.txt'))
		right_gripper = np.loadtxt(os.path.join(os.path.split(file)[0],'right_gripper.txt'))

		orig_left_traj = np.load(osp.join(osp.split(file)[0], 'Left_EE.npy'))
		orig_right_traj = np.load(osp.join(osp.split(file)[0], 'Right_EE.npy'))        

		joint_angle_trajectory = []
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
		elem['left_gripper'] = resample(left_gripper, n_samples)/100
		elem['right_gripper'] = resample(right_gripper, n_samples)/100
		elem['path_prefix'] = os.path.split(self.filelist[index])[0]
		elem['ra_trajectory'] = select_baxter_angles(elem['joint_angle_trajectory'], self.joint_names, arm='right')
		elem['la_trajectory'] = select_baxter_angles(elem['joint_angle_trajectory'], self.joint_names, arm='left')
		# If max norm of differences is <1.0, valid. 
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
