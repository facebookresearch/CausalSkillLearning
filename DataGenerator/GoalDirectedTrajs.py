# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from IPython import embed
import matplotlib.pyplot as plt

number_datapoints = 1
# number_datapoints = 50000
number_timesteps = 25

x_array_dataset = np.zeros((number_datapoints, number_timesteps, 2))
a_array_dataset = np.zeros((number_datapoints, number_timesteps-1, 2))
y_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
b_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
goal_array_dataset = np.zeros((number_datapoints, 1),dtype=int)

action_map = np.array([[0,-1],[-1,0],[0,1],[1,0]])
start_states = np.array([[-2,-2],[-2,2],[2,-2],[2,2]])*5
goal_states = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*5

valid_options = np.array([[2,3],[3,0],[1,2],[0,1]])

for i in range(number_datapoints):

	if i%1000==0:
		print("Processing Datapoint: ",i)

	# b_array_dataset[i,0] = 1.	
	goal_array_dataset[i] = np.random.random_integers(0,high=3)

	# Adding random noise to start state.
	x_array_dataset[i,-1] = goal_states[goal_array_dataset[i]] + 0.1*(np.random.random(2)-0.5)	
	goal = goal_states[goal_array_dataset[i]]

	reset_counter = 0
	# for t in range(number_timesteps-1):
	for t in reversed(range(number_timesteps-1)):

		# GET B	# Must end on b==0. 
		if t<(number_timesteps-2):
			# b_array[t] = np.random.binomial(1,prob_b_given_x)
			# b_array_dataset[i,t] = np.random.binomial(1,pb_x[0,x_array_dataset[i,t]])

			# If 3,4,5 timesteps have passed, terminate. 
			if t<3:
				b_array_dataset[i,t] = 0
			elif reset_counter>=3 and reset_counter<5:
				b_array_dataset[i,t] = np.random.binomial(1,0.33)
			elif reset_counter==5:
				b_array_dataset[i,t] = 1				
		elif t==(number_timesteps-2):
			b_array_dataset[i,t] = 1

		# GET Y
		if b_array_dataset[i,t]:
			current_state = x_array_dataset[i,t+1]
			# directions = current_state-goal.squeeze(0)
			directions = goal.squeeze(0)-current_state
			norm_directions = directions/abs(directions)

			# # Set valid options. 
			dot_product = np.dot(action_map, norm_directions)
			# valid_options = np.where(dot_product>=0)[0]
			# # Sincer we're going backwards in time, 
			valid_options = np.where(dot_product<=0)[0]

			# # axes = -goal/abs(goal)
			# # step1 = 30*np.ones((2))-axes*np.abs(x_array_dataset[i,t]-x_array_dataset[i,0])
			# # # baseline = t*20*np.sqrt(2)/20
			# # baseline = t
			# # step2 = step1-baseline
			# # step3 = step2/step2.sum()
			# # y_array_dataset[i,t] = np.random.choice(valid_options[goal_array_dataset[i][0]])
			# embed()
			dot_product = np.dot(action_map,directions)

			y_array_dataset[i,t] = np.argmax(dot_product)
			# y_array_dataset[i,t] = np.random.choice(valid_options)

			reset_counter = 0
		else:
			reset_counter+=1
			y_array_dataset[i,t] = y_array_dataset[i,t+1]

		# GET A
		a_array_dataset[i,t] = action_map[y_array_dataset[i,t]]-0.05+0.1*np.random.random((2))  		

		# GET X
		# x_array_dataset[i,t+1] = x_array_dataset[i,t]+a_array_dataset[i,t]
		x_array_dataset[i,t] = x_array_dataset[i,t+1]-a_array_dataset[i,t]

	plt.scatter(goal_states[:,0],goal_states[:,1],s=50)
	plt.scatter(x_array_dataset[i,:,0],x_array_dataset[i,:,1],cmap='jet',c=range(25))
	plt.xlim(-25,25)
	plt.ylim(-25,25)
	plt.show()

	# Roll over b's.
	b_array_dataset = np.roll(b_array_dataset,1,axis=1)
	

np.save("X_goal_directed.npy",x_array_dataset)
np.save("Y_goal_directed.npy",y_array_dataset)
np.save("B_goal_directed.npy",b_array_dataset)
np.save("A_goal_directed.npy",a_array_dataset)
np.save("G_goal_directed.npy",goal_array_dataset)
