#!/usr/bin/env python
# from headers import *
import numpy as np
from IPython import embed

number_datapoints = 50000
number_timesteps = 20

x_array_dataset = np.zeros((number_datapoints, number_timesteps, 2))
a_array_dataset = np.zeros((number_datapoints, number_timesteps-1, 2))
y_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
b_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)

action_map = np.array([[0,-1],[-1,0],[0,1],[1,0]])

for i in range(number_datapoints):
	if i%1000==0:
		print("Processing Datapoint: ",i)
	b_array_dataset[i,0] = 1.

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
			y_array_dataset[i,t] = np.random.random_integers(0,high=3)
			reset_counter = 0
		else:
			reset_counter+=1
			y_array_dataset[i,t] = y_array_dataset[i,t-1]

		# GET A
		a_array_dataset[i,t] = action_map[y_array_dataset[i,t]]-0.05+0.1*np.random.random((2))  		

		# GET X
		x_array_dataset[i,t+1] = x_array_dataset[i,t]+a_array_dataset[i,t]

	# embed()

np.save("X_array_continuous.npy",x_array_dataset)
np.save("Y_array_continuous.npy",y_array_dataset)
np.save("B_array_continuous.npy",b_array_dataset)
np.save("A_array_continuous.npy",a_array_dataset)