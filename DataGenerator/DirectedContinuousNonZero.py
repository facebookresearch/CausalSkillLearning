# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from IPython import embed

number_datapoints = 50000
number_timesteps = 25

x_array_dataset = np.zeros((number_datapoints, number_timesteps, 2))
a_array_dataset = np.zeros((number_datapoints, number_timesteps-1, 2))
y_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
b_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
goal_array_dataset = np.zeros((number_datapoints, 1),dtype=int)

action_map = np.array([[0,-1],[-1,0],[0,1],[1,0]])
start_states = np.array([[-2,-2],[-2,2],[2,-2],[2,2]])*5
valid_options = np.array([[2,3],[3,0],[1,2],[0,1]])

for i in range(number_datapoints):

	if i%1000==0:
		print("Processing Datapoint: ",i)
	b_array_dataset[i,0] = 1.

	# Select one of four starting points. (-2,-2), (-2,2), (2,-2), (2,2)
	goal_array_dataset[i] = np.random.random_integers(0,high=3)
	# Adding random noise to start state.
	x_array_dataset[i,0] = start_states[goal_array_dataset[i]] + 0.2*(np.random.random(2)-0.5)	
	goal = -start_states[goal_array_dataset[i]]

	reset_counter = 0
	for t in range(number_timesteps-1):

		# GET B
		if t>0:
			# b_array[t] = np.random.binomial(1,prob_b_given_x)
			# b_array_dataset[i,t] = np.random.binomial(1,pb_x[0,x_array_dataset[i,t]])

			# If 3,4,5 timesteps have passed, terminate. 
			if reset_counter>=3 and reset_counter<5:
				b_array_dataset[i,t] = np.random.binomial(1,0.33)
			elif reset_counter==5:
				b_array_dataset[i,t] = 1

		# GET Y
		if b_array_dataset[i,t]:			

			axes = -goal/abs(goal)
			step1 = 30*np.ones((2))-axes*np.abs(x_array_dataset[i,t]-x_array_dataset[i,0])
			# baseline = t*20*np.sqrt(2)/20
			baseline = t
			step2 = step1-baseline
			step3 = step2/step2.sum()
			y_array_dataset[i,t] = np.random.choice(valid_options[goal_array_dataset[i][0]])

			reset_counter = 0
		else:
			reset_counter+=1
			y_array_dataset[i,t] = y_array_dataset[i,t-1]

		# GET A
		a_array_dataset[i,t] = action_map[y_array_dataset[i,t]]-0.05+0.1*np.random.random((2))  		

		# GET X
		x_array_dataset[i,t+1] = x_array_dataset[i,t]+a_array_dataset[i,t]

np.save("X_dir_cont_nonzero.npy",x_array_dataset)
np.save("Y_dir_cont_nonzero.npy",y_array_dataset)
np.save("B_dir_cont_nonzero.npy",b_array_dataset)
np.save("A_dir_cont_nonzero.npy",a_array_dataset)
np.save("G_dir_cont_nonzero.npy",goal_array_dataset)
