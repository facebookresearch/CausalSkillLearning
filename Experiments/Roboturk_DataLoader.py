#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

class Roboturk_Dataset(Dataset):

	# LINK TO DATASET and INFO: http://roboturk.stanford.edu/dataset.html

	# Class implementing instance of Roboturk dataset. 
	def __init__(self):
		self.dataset_directory = '/checkpoint/tanmayshankar/Roboturk/RoboTurkPilot'

		# Require a task list. 

		# The task name is needed for setting the environment, rendering. 
		# We shouldn't need the environment for .. training though, should we? 

		self.task_list = ["bins-Bread", "bins-Can", "bins-Cereal", "bins-full", "bins-Milk", "pegs-full", "pegs-RoundNut", "pegs-SquareNut"]
		self.num_demos = np.array([1069, 1069, 1069, 1069, 1069, 1145, 1144, 1145])
		self.cummulative_num_demos = self.num_demos.cumsum()
		self.cummulative_num_demos = np.insert(self.cummulative_num_demos,0,0)
		# Append -1 to the start of cummulative_num_demos. This has two purposes. 
		# The first is that when we are at index 0 of the dataset, if we appended 0, np.searchsorted returns 0, rather than 1. 
		# For index 1, it returns 1. This was becoming inconsistent behavior for demonstrations in the same task. 
		# Now with -1 added to cumm_num_demos, when we are at task index 0, it would add -1 to the demo index. This is necessary for ALL tasks, not just the first...  
		# So that foils our really clever idea. 
		# Well, if the searchsorted returns the index of the equalling element, it probably consistently does this irrespective of vlaue. 
		# This means we can use this...

		# No need for a clever solution, searchsorted has a "side" option that takes care of this. 

		self.total_length = self.num_demos.sum()		

		# Load data from all tasks. 			
		self.files = []
		for i in range(len(self.task_list)):
			self.files.append(h5py.File("{0}/{1}/demo.hdf5".format(self.dataset_directory,self.task_list[i]),'r'))

		# Seems to follow joint angles order:
		# ('time','right_j0', 'head_pan', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6', 'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint', 'Milk0', 'Bread0', 'Cereal0', 'Can0').
		# Extract these into... 
		self.joint_angle_indices = [1,3,4,5,6,7,8]
		self.gripper_indices = [9,10]	
		self.ds_freq = 20
		# self.r_gripper_r_finger_joint = np.array([-0.0116,   0.020833])
		# self.r_gripper_l_finger_joint = np.array([-0.020833, 0.0135])

		# [l,r]
        # gripper_open = [0.0115, -0.0115]
        # gripper_closed = [-0.020833, 0.020833]
        

	def __len__(self):

		return self.total_length

	def __getitem__(self, index):

		if index>=self.total_length:
			print("Out of bounds of dataset.")
			return None

		# Get bucket that index falls into based on num_demos array. 
		task_index = np.searchsorted(self.cummulative_num_demos, index, side='right')-1
		
		if index==self.total_length-1:
			task_index-=1

		# Decide task ID, and new index modulo num_demos.
		# Subtract number of demonstrations in cumsum until then, and then 				
		new_index = index-self.cummulative_num_demos[max(task_index,0)]+1
		
		try:
			# Get raw state sequence. 
			state_sequence = self.files[task_index]['data/demo_{0}/states'.format(new_index)].value
		except:
			# If this failed, return invalid. 
			data_element = {}
			data_element['is_valid'] = False

			return data_element

		# Get joint angles from this state sequence.
		joint_values = state_sequence[:,self.joint_angle_indices]
		# Get gripper values from state sequence. 
		gripper_finger_values = state_sequence[:,self.gripper_indices]

		# Normalize gripper values. 

		# 1 is right finger. 0 is left finger. 
		# 1-0 is right-left. 
		
		gripper_values = gripper_finger_values[:,1]-gripper_finger_values[:,0]
		gripper_values = (gripper_values-gripper_values.min()) / (gripper_values.max()-gripper_values.min())
		gripper_values = 2*gripper_values-1

		concatenated_demonstration = np.concatenate([joint_values,gripper_values.reshape((-1,1))],axis=1)
		donwsampled_demonstration = resample(concatenated_demonstration, concatenated_demonstration.shape[0]//self.ds_freq)

		data_element = {}
		data_element['demo'] = donwsampled_demonstration
		# Trivially setting is valid to true until we come up wiuth a better strategy. 
		data_element['is_valid'] = True

		return data_element

	def close(self):
		for file in self.files:
			file.close()