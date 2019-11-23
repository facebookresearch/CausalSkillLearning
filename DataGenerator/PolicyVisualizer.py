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
# action_map = np.array([[-1,0],[0,-1],[1,0],[0,1]])

# start_states = np.array([[-2,-2],[-2,2],[2,-2],[2,2]])*5
goal_states = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*5

# Creating a policy map. 
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

policy_map = np.transpose(policy_map)


# x = np.meshgrid(range(9),range(9))
x = np.meshgrid(np.arange(9),np.arange(9))
dxdy = action_map[policy_map[x[0],x[1]]]

traj = np.zeros((10,2))
traj[0] = [0,8]
for t in range(9):
	# embed()
	action_index = policy_map[int(traj[t,0]),int(traj[t,1])]
	action = action_map[action_index]
	traj[t+1] = traj[t] + action
	print(action_index, action)

plt.ylim(9,-1)
plt.plot(traj[:,0],traj[:,1],'or')      
plt.plot(traj[:,0],traj[:,1],'r')      

plt.scatter(x[0],x[1])      
for i in range(9):                      
	for j in range(9):
		plt.arrow(x[0][i,j],x[1][i,j],0.1*dxdy[i,j,0],0.1*dxdy[i,j,1],width=0.01)

plt.show()

# embed()

# Transformed vis.
size = 9
scale = 5
scaled_size = scale*size
# policy_map = np.flipud(np.transpose(policy_map))
policy_map = np.transpose(policy_map)
# goal_based_policy_maps = np.zeros((4,size,size),dtype=int)
# goal_based_policy_maps[0] = copy.deepcopy(policy_map)
# goal_based_policy_maps[1] = np.rot90(policy_map)
# goal_based_policy_maps[2] = np.rot90(policy_map,k=2)
# goal_based_policy_maps[3] = np.rot90(policy_map,k=3)

def get_bucket(state, reference_state):
	# baseline = 4*np.ones(2)
	baseline = np.zeros(2)
	compensated_state = state - reference_state
	# compensated_state = (np.round(state - reference_state) + baseline).astype(int)

	scaled_size = scale*size
	x = (np.arange(-(size-1)/2,(size-1)/2+1)-0.5)*scale

	bucket = np.zeros((2))
	
	bucket[0] = min(np.searchsorted(x,compensated_state[0]),size-1)
	bucket[1] = min(np.searchsorted(x,compensated_state[1]),size-1)
	
	return bucket.astype(int)

goal_states = np.array([[-1,-1],[-1,1],[1,-1],[1,1]])*10

# goal_index = 1
# # meshrange = np.arange(-scaled_size/2,scaled_size/2+1,5)
# meshrange = (np.arange(-(size-1)/2,(size-1)/2+1)-0.5)*scale
# evalrange = (np.arange(-(size-1)/2,(size-1)/2+1)-1)*scale

# x = np.meshgrid(goal_states[goal_index,0]+meshrange,goal_states[goal_index,1]+meshrange) 

# dxdy = np.zeros((9,9,2))
# # dxdy = action_map[policy_map[x[0],x[1]]]
# plt.scatter(x[0],x[1])      
# plt.ylim(50,-50)

# arr = np.zeros((9,9,2))

# for i in range(9):                      
# 	for j in range(9):
# 		a = goal_states[goal_index,0]+evalrange[i]
# 		b = goal_states[goal_index,1]+evalrange[j]
# 		bucket = get_bucket(np.array([a,b]), goal_states[goal_index])
# 		arr[i,j,0] = i
# 		arr[i,j,1] = j
# 		dxdy[bucket[0],bucket[1]] = action_map[policy_map[bucket[0],bucket[1]]]
# 		plt.arrow(x[0][i,j],x[1][i,j],0.1*dxdy[i,j,0],0.1*dxdy[i,j,1],width=0.01*scale)

# plt.show()

for goal_index in range(4):
	# embed()
	# meshrange = np.arange(-scaled_size/2,scaled_size/2+1,5)
	meshrange = (np.arange(-(size-1)/2,(size-1)/2+1)-0.5)*scale
	evalrange = (np.arange(-(size-1)/2,(size-1)/2+1)-1)*scale

	x = np.meshgrid(goal_states[goal_index,0]+meshrange,goal_states[goal_index,1]+meshrange) 

	dxdy = np.zeros((9,9,2))
	# dxdy = action_map[policy_map[x[0],x[1]]]
	plt.scatter(x[0],x[1])      
	plt.ylim(50,-50)
	plt.xlim(-50,50)

	arr = np.zeros((9,9,2))

	for i in range(9):                      
		for j in range(9):
			a = goal_states[goal_index,0]+evalrange[i]
			b = goal_states[goal_index,1]+evalrange[j]
			bucket = get_bucket(np.array([a,b]), goal_states[goal_index])
			arr[i,j,0] = i
			arr[i,j,1] = j
			# dxdy[bucket[0],bucket[1]] = action_map[goal_based_policy_maps[goal_index,bucket[0],bucket[1]]]
			dxdy[bucket[0],bucket[1]] = action_map[policy_map[bucket[0],bucket[1]]]
			# plt.arrow(x[0][i,j],x[1][i,j],0.1*dxdy[i,j,0],0.1*dxdy[i,j,1],width=0.01*scale)
	
	# plt.quiver(x[0],x[1],0.1*dxdy[:,:,1],0.1*dxdy[:,:,0],width=0.0001,headwidth=4,headlength=2)			
	plt.quiver(x[0],x[1],0.1*dxdy[:,:,1],0.1*dxdy[:,:,0])

	traj_len = 20
	traj = np.zeros((20,2))
	traj[0] = np.random.randint(-25,high=25,size=2)
	
	for t in range(traj_len-1):
		
		bucket = get_bucket(traj[t], goal_states[goal_index])
		action_index = policy_map[bucket[0],bucket[1]]
		action = action_map[action_index]
		traj[t+1] = traj[t] + action
		
	plt.plot(traj[:,0],traj[:,1],'r')      
	plt.plot(traj[:,0],traj[:,1],'or')

	plt.show()

