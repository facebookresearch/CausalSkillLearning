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

		# FOR NOW: USE ONLY till 3200 images. 		
		return 3200
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
		image = cv2.imread(os.path.join(self.dataset_directory,"Image{0}.png".format(index)))
		coordinate_trajectory = np.load(os.path.join(self.dataset_directory,"Image{0}_Traj1.npy".format(index))).astype(float)

		action_sequence = self.parse_trajectory_actions(coordinate_trajectory)		

		return image, coordinate_trajectory, action_sequence

class SmallMapsDataset(Dataset):

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

class ToyDataset(Dataset):

	# Class implementing instance of dataset class for toy data. 

	def __init__(self, dataset_directory):
		self.dataset_directory = dataset_directory
		# For us, this is Research/Code/GraphPlanningNetworks/scripts/DatasetPlanning/CreateDemos/Demos2

		self.x_path = os.path.join(self.dataset_directory,"X_array_actions.npy")
		self.a_path = os.path.join(self.dataset_directory,"A_array_actions.npy")

		self.X_array = np.load(self.x_path)
		self.A_array = np.load(self.a_path)

	def __len__(self):
		return 50000

	def __getitem__(self, index):

		# Return trajectory and action sequence.
		return self.X_array[index],self.A_array[index]

class ContinuousToyDataset(Dataset):

	# Class implementing instance of dataset class for toy data. 

	def __init__(self, dataset_directory):
		self.dataset_directory = dataset_directory
		# For us, this is Research/Code/GraphPlanningNetworks/scripts/DatasetPlanning/CreateDemos/Demos2

		self.x_path = os.path.join(self.dataset_directory,"X_array_continuous.npy")
		self.a_path = os.path.join(self.dataset_directory,"A_array_continuous.npy")
		self.y_path = os.path.join(self.dataset_directory,"Y_array_continuous.npy")
		self.b_path = os.path.join(self.dataset_directory,"B_array_continuous.npy")

		self.X_array = np.load(self.x_path)
		self.A_array = np.load(self.a_path)
		self.Y_array = np.load(self.y_path)
		self.B_array = np.load(self.b_path)

	def __len__(self):
		return 50000

	def __getitem__(self, index):

		# Return trajectory and action sequence.
		return self.X_array[index],self.A_array[index]

	def get_latent_variables(self, index):
		return self.B_array[index],self.Y_array[index]

class ContinuousDirectedToyDataset(Dataset):

	# Class implementing instance of dataset class for toy data. 

	def __init__(self, dataset_directory):
		self.dataset_directory = dataset_directory
		# For us, this is Research/Code/GraphPlanningNetworks/scripts/DatasetPlanning/CreateDemos/Demos2

		self.x_path = os.path.join(self.dataset_directory,"X_array_directed_continuous.npy")
		self.a_path = os.path.join(self.dataset_directory,"A_array_directed_continuous.npy")
		self.y_path = os.path.join(self.dataset_directory,"Y_array_directed_continuous.npy")
		self.b_path = os.path.join(self.dataset_directory,"B_array_directed_continuous.npy")

		self.X_array = np.load(self.x_path)
		self.A_array = np.load(self.a_path)
		self.Y_array = np.load(self.y_path)
		self.B_array = np.load(self.b_path)

	def __len__(self):
		return 50000

	def __getitem__(self, index):

		# Return trajectory and action sequence.
		return self.X_array[index],self.A_array[index]

	def get_latent_variables(self, index):
		return self.B_array[index],self.Y_array[index]

class ContinuousNonZeroToyDataset(Dataset):

	# Class implementing instance of dataset class for toy data. 

	def __init__(self, dataset_directory):
		self.dataset_directory = dataset_directory
		# For us, this is Research/Code/GraphPlanningNetworks/scripts/DatasetPlanning/CreateDemos/Demos2

		self.x_path = os.path.join(self.dataset_directory,"X_array_continuous_nonzero.npy")
		self.a_path = os.path.join(self.dataset_directory,"A_array_continuous_nonzero.npy")
		self.y_path = os.path.join(self.dataset_directory,"Y_array_continuous_nonzero.npy")
		self.b_path = os.path.join(self.dataset_directory,"B_array_continuous_nonzero.npy")

		self.X_array = np.load(self.x_path)
		self.A_array = np.load(self.a_path)
		self.Y_array = np.load(self.y_path)
		self.B_array = np.load(self.b_path)

	def __len__(self):
		return 50000

	def __getitem__(self, index):

		# Return trajectory and action sequence.
		return self.X_array[index],self.A_array[index]

	def get_latent_variables(self, index):
		return self.B_array[index],self.Y_array[index]

class ContinuousDirectedNonZeroToyDataset(Dataset):

	# Class implementing instance of dataset class for toy data. 

	def __init__(self, dataset_directory):
		self.dataset_directory = dataset_directory
		# For us, this is Research/Code/GraphPlanningNetworks/scripts/DatasetPlanning/CreateDemos/Demos2

		self.x_path = os.path.join(self.dataset_directory,"X_dir_cont_nonzero.npy")
		self.a_path = os.path.join(self.dataset_directory,"A_dir_cont_nonzero.npy")
		self.y_path = os.path.join(self.dataset_directory,"Y_dir_cont_nonzero.npy")
		self.b_path = os.path.join(self.dataset_directory,"B_dir_cont_nonzero.npy")
		self.g_path = os.path.join(self.dataset_directory,"G_dir_cont_nonzero.npy")

		self.X_array = np.load(self.x_path)
		self.A_array = np.load(self.a_path)
		self.Y_array = np.load(self.y_path)
		self.B_array = np.load(self.b_path)
		self.G_array = np.load(self.g_path)

	def __len__(self):
		return 50000

	def __getitem__(self, index):

		# Return trajectory and action sequence.
		return self.X_array[index],self.A_array[index]

	def get_latent_variables(self, index):
		return self.B_array[index],self.Y_array[index]

class GoalDirectedDataset(Dataset):

	# Class implementing instance of dataset class for toy data. 

	def __init__(self, dataset_directory):
		self.dataset_directory = dataset_directory
		# For us, this is Research/Code/GraphPlanningNetworks/scripts/DatasetPlanning/CreateDemos/Demos2

		self.x_path = os.path.join(self.dataset_directory,"X_goal_directed.npy")
		self.a_path = os.path.join(self.dataset_directory,"A_goal_directed.npy")
		self.y_path = os.path.join(self.dataset_directory,"Y_goal_directed.npy")
		self.b_path = os.path.join(self.dataset_directory,"B_goal_directed.npy")
		self.g_path = os.path.join(self.dataset_directory,"G_goal_directed.npy")

		self.X_array = np.load(self.x_path)
		self.A_array = np.load(self.a_path)
		self.Y_array = np.load(self.y_path)
		self.B_array = np.load(self.b_path)
		self.G_array = np.load(self.g_path)

	def __len__(self):
		return 50000

	def __getitem__(self, index):

		# Return trajectory and action sequence.
		return self.X_array[index],self.A_array[index]

	def get_latent_variables(self, index):
		return self.B_array[index],self.Y_array[index]

	def get_goal(self, index):
		return self.G_array[index]

class DeterministicGoalDirectedDataset(Dataset):

	# Class implementing instance of dataset class for toy data. 

	def __init__(self, dataset_directory):
		self.dataset_directory = dataset_directory
		# For us, this is Research/Code/GraphPlanningNetworks/scripts/DatasetPlanning/CreateDemos/Demos2

		self.x_path = os.path.join(self.dataset_directory,"X_deter_goal_directed.npy")
		self.a_path = os.path.join(self.dataset_directory,"A_deter_goal_directed.npy")
		self.y_path = os.path.join(self.dataset_directory,"Y_deter_goal_directed.npy")
		self.b_path = os.path.join(self.dataset_directory,"B_deter_goal_directed.npy")
		self.g_path = os.path.join(self.dataset_directory,"G_deter_goal_directed.npy")

		self.X_array = np.load(self.x_path)
		self.A_array = np.load(self.a_path)
		self.Y_array = np.load(self.y_path)
		self.B_array = np.load(self.b_path)
		self.G_array = np.load(self.g_path)

		self.goal_states = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*5

	def __len__(self):
		return 50000

	def __getitem__(self, index):

		# Return trajectory and action sequence.
		return self.X_array[index],self.A_array[index]

	def get_latent_variables(self, index):
		return self.B_array[index],self.Y_array[index]

	def get_goal(self, index):
		return self.G_array[index]

	def get_goal_position(self, index):
		return self.goal_states[self.G_array[index]]

class SeparableDataset(Dataset):

	# Class implementing instance of dataset class for toy data. 

	def __init__(self, dataset_directory):
		self.dataset_directory = dataset_directory
		# For us, this is Research/Code/GraphPlanningNetworks/scripts/DatasetPlanning/CreateDemos/Demos2

		self.x_path = os.path.join(self.dataset_directory,"X_separable.npy")
		self.a_path = os.path.join(self.dataset_directory,"A_separable.npy")
		self.y_path = os.path.join(self.dataset_directory,"Y_separable.npy")
		self.b_path = os.path.join(self.dataset_directory,"B_separable.npy")
		self.g_path = os.path.join(self.dataset_directory,"G_separable.npy")
		self.s_path = os.path.join(self.dataset_directory,"StartConfig_separable.npy")

		self.X_array = np.load(self.x_path)
		self.A_array = np.load(self.a_path)
		self.Y_array = np.load(self.y_path)
		self.B_array = np.load(self.b_path)
		self.G_array = np.load(self.g_path)
		self.S_array = np.load(self.s_path)

	def __len__(self):
		return 50000

	def __getitem__(self, index):

		# Return trajectory and action sequence.
		return self.X_array[index],self.A_array[index]

	def get_latent_variables(self, index):
		return self.B_array[index],self.Y_array[index]

	def get_goal(self, index):
		return self.G_array[index]

	def get_startconfig(self, index):
		return self.S_array[index]