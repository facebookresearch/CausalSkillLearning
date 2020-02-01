import numpy as np, glob
from IPython import embed

# Evaluate baselineRL methods. 
a = 86
b = 86
prefix = 'RL'

for i in range(a,b+1):

	model_template = "RL{0}/saved_models/Model_epoch*".format(i)
	models = glob.glob(model_template)	
	# number_models = [int((model.lstrip("RL{0}/saved_models/Model_epoch".format(i))).zfill(4)) for model in models]
	max_model = int(models[-1].lstrip("RL{0}/saved_models/Model_epoch".format(i)))

	model_range = np.arange(0,max_model+100,100)
	rewards = np.zeros((len(model_range)))

	for j in range(len(model_range)):
		rewards[j] = np.load("RL{0}/MEval/m{1}/Mean_Reward_RL{0}.npy".format(i,model_range[j]))

embed()
# x = np.arange(0,260,20)
# dists = np.zeros((6,len(x),100))
# a = 6
# b = 12
# for i in range(a,b):
# 	for j in range(len(x)):
# 		dists[i-a,j] = np.load("IL0{0}/MEval/m{1}/Total_Rewards_IL0{0}.npy".format(str(i).zfill(2),x[j]))


# IL 
a = 18
b = 23
prefix = 'IL0'
increment = 20
reward_list = []

for i in range(a,b+1):

	model_template = "{0}{1}/saved_models/Model_epoch*".format(prefix,i)
	models = glob.glob(model_template)	
	# number_models = [int((model.lstrip("RL{0}/saved_models/Model_epoch".format(i))).zfill(4)) for model in models]
	max_model = int(models[-1].lstrip("{0}{1}/saved_models/Model_epoch".format(prefix,i)))

	model_range = np.arange(0,max_model+increment,increment)
	rewards = np.zeros((len(model_range)))

	for j in range(len(model_range)):
		rewards[j] = np.load("{2}{0}/MEval/m{1}/Mean_Reward_{2}{0}.npy".format(i,model_range[j],prefix))

	reward_list.append(rewards)

# Get distances
a = 30
b = 37
prefix = 'RJ'
increment = 20
distance_list = []

for i in range(a,b+1):

	model_template = "{0}{1}/saved_models/Model_epoch*".format(prefix,i)
	models = glob.glob(model_template)	
	# number_models = [int((model.lstrip("RL{0}/saved_models/Model_epoch".format(i))).zfill(4)) for model in models]
	max_model = int(models[-1].lstrip("{0}{1}/saved_models/Model_epoch".format(prefix,i)))
	max_model = max_model-max_model%increment
	model_range = np.arange(0,max_model+increment,increment)
	distances = np.zeros((len(model_range)))

	for j in range(len(model_range)):
		distances[j] = np.load("{2}{0}/MEval/m{1}/Mean_Trajectory_Distance_{2}{0}.npy".format(i,model_range[j],prefix))

	distance_list.append(distances)
