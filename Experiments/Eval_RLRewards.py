import numpy as np, glob, os
from IPython import embed

# Env list. 
environment_names = ["SawyerPickPlaceBread","SawyerPickPlaceCan","SawyerPickPlaceCereal","SawyerPickPlaceMilk","SawyerNutAssemblyRound","SawyerNutAssemblySquare"]

# Evaluate baselineRL methods. 
a = 86
b = 86

a = 130
b = 137
prefix = 'RL'
increment = 100
reward_list = []

for i in range(a,b+1):

	model_template = "RL{0}/saved_models/Model_epoch*".format(i)
	models = glob.glob(model_template)	
	# number_models = [int((model.lstrip("RL{0}/saved_models/Model_epoch".format(i))).zfill(4)) for model in models]
	max_model = int(models[-1].lstrip("RL{0}/saved_models/Model_epoch".format(i)))

	model_range = np.arange(0,max_model+increment,increment)
	rewards = np.zeros((len(model_range)))

	for j in range(len(model_range)):
		rewards[j] = np.load("RL{0}/MEval/m{1}/Mean_Reward_RL{0}.npy".format(i,model_range[j]))

	reward_list.append(rewards)

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

################################################
# Env list. 
environment_names = ["SawyerPickPlaceBread","SawyerPickPlaceCan","SawyerPickPlaceCereal","SawyerPickPlaceMilk","SawyerNutAssemblyRound","SawyerNutAssemblySquare"]

# Evaluate baselineRL methods. 
a = 5
b = 12
prefix = 'downRL'
increment = 20
reward_list = []

for i in range(a,b+1):

	padded_index = str(i).zfill(3)

	model_template = "{1}{0}/saved_models/Model_epoch*".format(padded_index,prefix)
	models = glob.glob(model_template)	
	# number_models = [int((model.lstrip("RL{0}/saved_models/Model_epoch".format(i))).zfill(4)) for model in models]
	max_model = int(models[-1].lstrip("{1}{0}/saved_models/Model_epoch".format(padded_index,prefix)))
	max_model = max_model-max_model%increment
	model_range = np.arange(0,max_model+increment,increment)
	rewards = np.zeros((len(model_range)))

	for j in range(len(model_range)):
		rewards[j] = np.load("{2}{0}/MEval/m{1}/Mean_Reward_{2}{0}.npy".format(padded_index,model_range[j],prefix))
		# rewards[j] = np.load("{0}{1}/MEval/m{2}/Mean_Reward_{0}{1}.npy".format(prefix,padded_indexi,model_range[j],prefix))
	reward_list.append(rewards)

##############################################
# MOcap distances

# Get distances
a = 1
b = 2
prefix = 'Mocap00'
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