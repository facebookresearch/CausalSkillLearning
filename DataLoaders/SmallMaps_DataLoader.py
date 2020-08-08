# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from headers import *

class GridWorldDataset(Dataset):

	# Class implementing instance of dataset class for gridworld data. 

	def __init__(self, dataset_directory):
		self.dataset_directory = dataset_directory
		# For us, this is Research/Code/GraphPlanningNetworks/scripts/DatasetPlanning/CreateDemos/Demos2

		self.action_map = np.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])
		## UP, DOWN, LEFT, RIGHT, UPLEFT, UPRIGHT, DOWNLEFT, DOWNRIGHT. ##

	def __len__(self):

		# Find out how many images we've stored. 
		filelist = glob.glob(os.path.join(self.dataset_directory,"*.png"))		
		return 4000
		# return len(filelist)		

	def parse_trajectory_actions(self, coordinate_trajectory):
		# Takes coordinate trajectory, returns action index taken. 

		state_diffs = np.diff(coordinate_trajectory,axis=0)
		action_sequence = np.zeros((len(state_diffs)),dtype=int)

		for i in range(len(state_diffs)):
			for k in range(len(self.action_map)):
				if (state_diffs[i]==self.action_map[k]).all():
					action_sequence[i]=k

		return action_sequence.astype(float)

	def __getitem__(self, index):

		# The getitem function must return a Map-Trajectory pair. 
		# We will handle per-timestep processes within our code. 
		# Assumes index is within range [0,len(filelist)-1]
		image = np.load(os.path.join(self.dataset_directory,"Map{0}.npy".format(index)))
		time_limit = 20
		coordinate_trajectory = np.load(os.path.join(self.dataset_directory,"Map{0}_Traj1.npy".format(index))).astype(float)[:time_limit]
		action_sequence = self.parse_trajectory_actions(coordinate_trajectory)		

		return image, coordinate_trajectory, action_sequence