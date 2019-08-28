#!/usr/bin/env python
from headers import *

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