# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from IPython import embed
import matplotlib.pyplot as plt

# number_datapoints = 20
number_datapoints = 50000
number_timesteps = 20

x_array_dataset = np.zeros((number_datapoints, number_timesteps, 2))
a_array_dataset = np.zeros((number_datapoints, number_timesteps-1, 2))
y_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
b_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
goal_array_dataset = np.zeros((number_datapoints, 1),dtype=int)
start_config_dataset = np.zeros((number_datapoints, 1),dtype=int)

action_map = np.array([[0,-1],[-1,0],[0,1],[1,0]])
start_scale = 15
start_states = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*start_scale
goal_states = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*5
scale = 5
start_configs = np.zeros((4,5,2),dtype=int)
start_configs[[0,3]] = np.array([[-2,2],[-1,1],[0,0],[1,-1],[2,-2]])*scale
start_configs[[1,2]] = np.array([[-2,-2],[-1,-1],[0,0],[1,1],[2,2]])*scale

# valid_options = np.array([[2,3],[3,0],[1,2],[0,1]])
valid_options = np.array([[3,2],[3,0],[2,1],[0,1]])
lim = 50

progression_of_options = np.zeros((5,4),dtype=int)
progression_of_options[1,0] = 1
progression_of_options[2,:2] = 1
progression_of_options[3,1:] = 1
progression_of_options[4,:] = 1

for i in range(number_datapoints):

	if i%1000==0:
		print("Processing Datapoint: ",i)

	goal_array_dataset[i] = np.random.random_integers(0,high=3)
	start_config_dataset[i] = np.random.random_integers(0,high=4)
	# start_config_dataset[i] = 4
	
	# Adding random noise to start state.
	x_array_dataset[i,0] = start_states[goal_array_dataset[i]] + start_configs[goal_array_dataset[i],start_config_dataset[i]] + 0.1*(np.random.random(2)-0.5)	

	reset_counter = 0
	option_counter = 0

	for t in range(number_timesteps-1):

		# GET B
		if t==0:
			b_array_dataset[i,t] = 1
		if t>0:
			# If 3,4,5 timesteps have passed, terminate. 
			if reset_counter>=3 and reset_counter<5:
				b_array_dataset[i,t] = np.random.binomial(1,0.33)
			elif reset_counter==5:
				b_array_dataset[i,t] = 1

		# GET Y
		if b_array_dataset[i,t]:
			current_state = x_array_dataset[i,t]
			
			# select new y_array_dataset[i,t]
			y_array_dataset[i,t] = valid_options[goal_array_dataset[i]][0][progression_of_options[start_config_dataset[i],min(option_counter,3)]]

			option_counter+=1
			reset_counter = 0			
		else:
			reset_counter+=1
			y_array_dataset[i,t] = y_array_dataset[i,t-1]

		# GET A
		a_array_dataset[i,t] = action_map[y_array_dataset[i,t]]+0.1*(np.random.random((2))-0.5)

		# GET X
		# Already taking care of backwards generation here, no need to use action_compliments. 

		x_array_dataset[i,t+1] = x_array_dataset[i,t]+a_array_dataset[i,t]	

	# plt.scatter(goal_states[:,0],goal_states[:,1],s=50)
	# # plt.scatter()
	# plt.scatter(x_array_dataset[i,:,0],x_array_dataset[i,:,1],cmap='jet',c=range(number_timesteps))
	# plt.xlim(-lim,lim)
	# plt.ylim(-lim,lim)
	# plt.show()


	# Roll over b's.
	b_array_dataset = np.roll(b_array_dataset,1,axis=1)
	

np.save("X_separable.npy",x_array_dataset)
np.save("Y_separable.npy",y_array_dataset)
np.save("B_separable.npy",b_array_dataset)
np.save("A_separable.npy",a_array_dataset)
np.save("G_separable.npy",goal_array_dataset)
np.save("StartConfig_separable.npy",start_config_dataset)
