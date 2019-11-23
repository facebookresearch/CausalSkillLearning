#!/usr/bin/env python
import numpy as np, copy
from IPython import embed
import matplotlib.pyplot as plt

number_datapoints = 20
# number_datapoints = 50000
number_timesteps = 25

x_array_dataset = np.zeros((number_datapoints, number_timesteps, 2))
a_array_dataset = np.zeros((number_datapoints, number_timesteps-1, 2))
y_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
b_array_dataset = np.zeros((number_datapoints, number_timesteps-1),dtype=int)
goal_array_dataset = np.zeros((number_datapoints, 1),dtype=int)

action_map = np.array([[0,-1],[-1,0],[0,1],[1,0]])
# start_states = np.array([[-2,-2],[-2,2],[2,-2],[2,2]])*5
goal_states = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*10

# Creating a policy map. 
lim = 50
size = 9
scale = 5
policy_map = np.zeros((size,size),dtype=int)

# Row wise assignment: 
policy_map[0,:] = 2

policy_map[1,:7] = 2
policy_map[1,7:] = 1

policy_map[2:4,0] = 2
policy_map[2:4,1:4] = 3
policy_map[2:4,4:7] = 2
policy_map[2:4,7:] = 1

policy_map[4,:4] = 3
policy_map[4,4] = 3
policy_map[4,5:] = 1

policy_map[5,:3] = 3
policy_map[5,3:5] = 0
policy_map[5,5:] = 1

policy_map[6,:2] = 3
policy_map[6,2:7] = 0
policy_map[6,7:] = 1

policy_map[7:,0] = 3
policy_map[7:,1:7] = 0
policy_map[7:,7:] = 1

# policy_map = np.transpose(policy_map)

goal_based_policy_maps = np.zeros((4,size,size))
goal_based_policy_maps[0] = copy.deepcopy(policy_map)
goal_based_policy_maps[1] = np.flipud(policy_map)
goal_based_policy_maps[2] = np.fliplr(policy_map)
goal_based_policy_maps[3] = np.flipud(np.fliplr(policy_map))

def get_bucket(state, reference_state):
	# baseline = 4*np.ones(2)
	baseline = np.zeros(2)
	compensated_state = state - reference_state
	# compensated_state = (np.round(state - reference_state) + baseline).astype(int)
	
	x = (np.arange(-(size-1)/2,(size-1)/2+1)-0.5)*scale
	
	bucket = np.zeros((2))
	
	bucket[0] = min(np.searchsorted(x,compensated_state[0]),size-1)
	bucket[1] = min(np.searchsorted(x,compensated_state[1]),size-1)
	
	return bucket.astype(int)

for i in range(number_datapoints):

	if i%1000==0:
		print("Processing Datapoint: ",i)

	# b_array_dataset[i,0] = 1.	
	goal_array_dataset[i] = np.random.random_integers(0,high=3)

	# Adding random noise to start state.
	# x_array_dataset[i,0] = goal_states[goal_array_dataset[i]] + 0.1*(np.random.random(2)-0.5)

	scale = 25
	x_array_dataset[i,0] = goal_states[goal_array_dataset[i]] + scale*(np.random.random(2)-0.5)
	goal = goal_states[goal_array_dataset[i]]

	reset_counter = 0
	for t in range(number_timesteps-1):

		# GET B
		if t>0:
			# If 3,4,5 timesteps have passed, terminate. 
			if reset_counter>=3 and reset_counter<5:
				b_array_dataset[i,t] = np.random.binomial(1,0.33)
			elif reset_counter==5:
				b_array_dataset[i,t] = 1

		# GET Y
		if b_array_dataset[i,t]:
			current_state = x_array_dataset[i,t]

			# Select options from policy map, based on the bucket the current state falls in. 
			bucket = get_bucket(current_state, goal_states[goal_array_dataset[i]][0])
			# Now that we've the bucket, pick the option we should be executing given the bucket.

			if (bucket==0).all():
				y_array_dataset[i,t] = np.random.randint(0,high=4)
			else:
				y_array_dataset[i,t] = goal_based_policy_maps[goal_array_dataset[i], bucket[0], bucket[1]]
				y_array_dataset[i,t] = policy_map[bucket[0], bucket[1]]
			reset_counter = 0
		else:
			reset_counter+=1
			y_array_dataset[i,t] = y_array_dataset[i,t-1]

		# GET A
		a_array_dataset[i,t] = action_map[y_array_dataset[i,t]]-0.1*(np.random.random((2))-0.5)

		# GET X
		# Already taking care of backwards generation here, no need to use action_compliments. 

		x_array_dataset[i,t+1] = x_array_dataset[i,t]+a_array_dataset[i,t]	

	plt.scatter(goal_states[:,0],goal_states[:,1],s=50)
	# plt.scatter()
	plt.scatter(x_array_dataset[i,:,0],x_array_dataset[i,:,1],cmap='jet',c=range(25))
	plt.xlim(-lim,lim)
	plt.ylim(-lim,lim)
	plt.show()

	# Roll over b's.
	b_array_dataset = np.roll(b_array_dataset,1,axis=1)
	

np.save("X_goal_directed.npy",x_array_dataset)
np.save("Y_goal_directed.npy",y_array_dataset)
np.save("B_goal_directed.npy",b_array_dataset)
np.save("A_goal_directed.npy",a_array_dataset)
np.save("G_goal_directed.npy",goal_array_dataset)







