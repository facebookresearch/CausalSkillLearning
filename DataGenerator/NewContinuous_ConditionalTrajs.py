#!/usr/bin/env python
from headers import *
import numpy as np
from IPython import embed

number_datapoints = 1
number_timesteps = 25
lim = 25

x_array_dataset = np.zeros((number_datapoints, number_timesteps, 2))
a_array_dataset = np.zeros((number_datapoints, number_timesteps-1, 2))
y_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
b_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
goal_array_dataset = np.zeros((number_datapoints, 1),dtype=int)
start_array_dataset = np.zeros((number_datapoints, 1),dtype=int)

action_map = np.array([[0,-1],[-1,0],[0,1],[1,0]])
start_states = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*5
goal_states = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*5
# valid_options = np.array([[2,3],[3,0],[1,2],[0,1]])

for i in range(number_datapoints):

	if i%1000==0:
		print("Processing Datapoint: ",i)
	b_array_dataset[i,0] = 1.

	# Select one of four starting points. (-2,-2), (-2,2), (2,-2), (2,2)
	start_array_dataset[i] = np.random.random_integers(0,high=3)
	goal_array_dataset[i] = np.random.random_integers(0,high=3)
	# Goal. 
	goal = goal_states[goal_array_dataset[i]]
	# Set start state as perturbed version of goal. 
	x_array_dataset[i,0] = goal+start_states[start_array_dataset[i]]
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

			# Measure distances to goal. 
			# directions = goal - x_array_dataset[i,t]
			# directions = directions/abs(directions)

			# # Set valid options. 
			# dot_product = np.dot(action_map, directions)
			# # valid_options = np.where(dot_product>=0)[0]
			# # Sincer we're going backwards in time, 
			# valid_options = np.where(dot_product>=0)[0]			

			# y_array_dataset[i,t] = np.random.choice(valid_options[goal_array_dataset[i][0]])

			current_state = x_array_dataset[i,t]
			unnorm_directions = goal.squeeze(0) - current_state
			directions = unnorm_directions/abs(unnorm_directions)

			# Set valid options. 
			dot_product = np.dot(action_map, unnorm_directions)
			y_array_dataset[i,t] = np.argmax(dot_product)

			reset_counter = 0
		else:
			reset_counter+=1
			y_array_dataset[i,t] = y_array_dataset[i,t-1]

		# GET A
		a_array_dataset[i,t] = action_map[y_array_dataset[i,t]]-0.05+0.1*np.random.random((2))  		

		# GET X
		x_array_dataset[i,t+1] = x_array_dataset[i,t]+a_array_dataset[i,t]

	plt.scatter(goal_states[:,0],goal_states[:,1],s=50)
	plt.scatter(x_array_dataset[i,:,0],x_array_dataset[i,:,1],cmap='jet',c=range(25))
	plt.xlim(-lim, lim)
	plt.ylim(-lim, lim)
	plt.show()

np.save("X_array_newcont_cond.npy",x_array_dataset)
np.save("Y_array_newcont_cond.npy",y_array_dataset)
np.save("B_array_newcont_cond.npy",b_array_dataset)
np.save("A_array_newcont_cond.npy",a_array_dataset)
np.save("G_array_newcont_cond.npy",goal_array_dataset)
np.save("S_array_newcont_cond.npy",start_array_dataset)