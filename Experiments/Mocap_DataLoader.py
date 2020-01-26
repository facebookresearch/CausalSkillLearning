#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from headers import *

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

class Mocap_Dataset(Dataset):

	def __init__(self, split='all'):
		self.dataset_directory = '/checkpoint/tanmayshankar/Mocap/'

		# Load the entire set of trajectories. 
		self.data_list = np.load(os.path.join(self.dataset_directory, "Demo_Array.npy"),allow_pickle=True)

		self.dataset_length = len(self.data_list)
	
	def __len__(self):
		# Return length of file list. 
		return self.dataset_length

	def __getitem__(self, index):
		# Return n'th item of dataset.
		# This has already processed everything.

		# Remember, the global and local posiitons are all stored as Number_Frames x Number_Joints x 3 array. 
		# Change this to # Number_Frames x Number_Dimensions...? But the dimensions are not independent.. so what do we do?

		return self.data_list[index]

	def compute_statistics(self):
		embed()