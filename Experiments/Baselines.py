#!/usr/bin/env python
from headers import *
from PolicyManager import *
import TFLogger

class Baselines(PolicyManager_BaseClass):

	def __init__(self):
		super(Baselines, self).__init__()

	def forward(self):
		pass

	# def get_reconstructed_trajectory(self, i):
	# 	pass

	# def evaluate_state_distances(self):

	# 	# Evaluate state distances between original and reconstructed trajectory across set of trajectories. 

	# 	# For every item in the epoch:
	# 	for i in range(len(self.dataset)):

	# 		print("Trajectory: ",i)		

	# 		if self.dataset['is_valid']:
	# 			reconstructed_trajectory = self.get_reconstructed_trajectory(i)

class LSTM_Baseline(PolicyManager_Pretrain):

	def __init__(self):
		super(LSTM_Baseline, self).__init__()

	def forward(self):
		pass

	def evaluate_state_distances(self):

		# For every item in the epoch:
		for i in range(len(self.dataset)):

			print("Trajectory: ",i)					
			sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)

	
