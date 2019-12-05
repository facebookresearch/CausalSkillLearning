#!/usr/bin/env python
from headers import *
import DataLoaders, MIME_DataLoader
# import PolicyManager_GTSubpolicy, PolicyManager_LearntSubpolicy, PolicyManager_BatchGTSubpolicy
import PolicyManager_LearntSubpolicy, Pretrain_Subpolicy, Old_Pretrain_Sub

class Master():

	def __init__(self, arguments):
		self.args = arguments 

		# Define Data Loader. 
		if self.args.data=='Bigmaps':
			self.dataset = DataLoaders.GridWorldDataset(self.args.datadir)
		elif self.args.data=='Smallmaps':
			self.dataset = DataLoaders.SmallMapsDataset(self.args.datadir)
		elif self.args.data=='ToyData':
			self.dataset = DataLoaders.ToyDataset(self.args.datadir)
		elif self.args.data=='Continuous':
			self.dataset = DataLoaders.ContinuousToyDataset(self.args.datadir)
		elif self.args.data=='ContinuousNonZero':
			self.dataset = DataLoaders.ContinuousNonZeroToyDataset(self.args.datadir)
		elif self.args.data=='ContinuousDir':			
			self.dataset = DataLoaders.ContinuousDirectedToyDataset(self.args.datadir)
		elif self.args.data=='ContinuousDirNZ':			
			self.dataset = DataLoaders.ContinuousDirectedNonZeroToyDataset(self.args.datadir)
		elif self.args.data=='GoalDirected':
			self.dataset = DataLoaders.GoalDirectedDataset(self.args.datadir)
		elif self.args.data=='DeterGoal':
			self.dataset = DataLoaders.DeterministicGoalDirectedDataset(self.args.datadir)			
		elif self.args.data=='Separable':
			self.dataset = DataLoaders.SeparableDataset(self.args.datadir)			
		elif self.args.data=='MIME':
			self.dataset = MIME_DataLoader.MIME_Dataset()

		# if self.args.setting=='gtsub':
		# 	self.policy_manager = PolicyManager_GTSubpolicy.PolicyManager(self.args.number_policies, self.dataset, self.args)
		# # if self.args.setting=='just_actions':
		# # 	self.policy_manager = PolicyManager_GTSubpolicy_ActionLikelihood.PolicyManager(self.args.number_policies, self.dataset, self.args)
		# elif self.args.setting=='batchgtsub':
		# 	self.policy_manager = PolicyManager_BatchGTSubpolicy.PolicyManager(self.args.number_policies, self.dataset, self.args)
		if self.args.setting=='learntsub':
			self.policy_manager = PolicyManager_LearntSubpolicy.PolicyManager(self.args.number_policies, self.dataset, self.args)
		elif self.args.setting=='pretrain_sub':
			self.policy_manager = Pretrain_Subpolicy.PolicyManager(self.args.number_policies, self.dataset, self.args)
		elif self.args.setting=='oldpretrain_sub':
			self.policy_manager = Old_Pretrain_Sub.PolicyManager(self.args.number_policies, self.dataset, self.args)

		# Create networks and training operations. 
		self.policy_manager.setup()

	def run(self):
		if self.args.setting=='pretrain_sub':			
			if self.args.train:
				if self.args.model:
					self.policy_manager.train(self.args.model)
				else:
					self.policy_manager.train()
			else:
				self.policy_manager.evaluate(self.args.model)
		else:
			if self.args.train:
				if self.args.model:
					self.policy_manager.train(self.args.model)
				else:
					if self.args.subpolicy_model:
						print("Just loading subpolicies.")
						self.policy_manager.load_all_models(self.args.subpolicy_model, just_subpolicy=True)
					self.policy_manager.train()
			else:
				# self.policy_manager.train(self.args.model)
				self.policy_manager.evaluate(self.args.model)

def parse_arguments():
	parser = argparse.ArgumentParser(description='Learning Skills from Demonstrations')

	# Setup training. 
	parser.add_argument('--datadir', dest='datadir',type=str,default='../DataGenerator/ContData/')
	parser.add_argument('--train',dest='train',type=int,default=0)
	parser.add_argument('--debug',dest='debug',type=int,default=0)
	parser.add_argument('--notes',dest='notes',type=str)
	parser.add_argument('--name',dest='name',type=str,default=None)
	parser.add_argument('--fake_batch_size',dest='fake_batch_size',type=int,default=1)
	parser.add_argument('--batch_size',dest='batch_size',type=int,default=1)
	parser.add_argument('--training_phase_size',dest='training_phase_size',type=int,default=500000)
	parser.add_argument('--data',dest='data',type=str,default='Continuous')
	parser.add_argument('--setting',dest='setting',type=str,default='gtsub')
	parser.add_argument('--model',dest='model',type=str)	
	parser.add_argument('--logdir',dest='logdir',type=str,default='Experiment_Logs/')

	# Training setting. 
	parser.add_argument('--discrete_z',dest='discrete_z',type=int,default=0)
	parser.add_argument('--z_dimensions',dest='z_dimensions',type=int,default=8)
	parser.add_argument('--condition_size',dest='condition_size',type=int,default=4)
	parser.add_argument('--new_gradient',dest='new_gradient',type=int,default=1)
	parser.add_argument('--b_prior',dest='b_prior',type=int,default=1)
	parser.add_argument('--reparam',dest='reparam',type=int,default=1)	
	parser.add_argument('--number_policies',dest='number_policies',type=int,default=4)
	parser.add_argument('--fix_subpolicy',dest='fix_subpolicy',type=int,default=1)
	parser.add_argument('--subpolicy_model',dest='subpolicy_model',type=str)
	parser.add_argument('--traj_length',dest='traj_length',type=int,default=10)
	parser.add_argument('--skill_length',dest='skill_length',type=int,default=5)
	parser.add_argument('--var_skill_length',dest='var_skill_length',type=int,default=0)
	parser.add_argument('--display_freq',dest='display_freq',type=int,default=10000)
	parser.add_argument('--save_freq',dest='save_freq',type=int,default=1)	
	parser.add_argument('--eval_freq',dest='eval_freq',type=int,default=20)	

	parser.add_argument('--entropy',dest='entropy',type=int,default=0)
	parser.add_argument('--var_entropy',dest='var_entropy',type=int,default=0)
	parser.add_argument('--ent_weight',dest='ent_weight',type=float,default=0.)
	parser.add_argument('--var_ent_weight',dest='var_ent_weight',type=float,default=2.)
	
	parser.add_argument('--z_ex_bias',dest='z_exploration_bias',type=float,default=0.)
	parser.add_argument('--b_ex_bias',dest='b_exploration_bias',type=float,default=0.)
	parser.add_argument('--lat_z_wt',dest='lat_z_wt',type=float,default=0.1)
	parser.add_argument('--lat_b_wt',dest='lat_b_wt',type=float,default=1.)
	parser.add_argument('--z_probability_factor',dest='z_probability_factor',type=float,default=0.1)
	parser.add_argument('--b_probability_factor',dest='b_probability_factor',type=float,default=0.1)
	parser.add_argument('--subpolicy_clamp_value',dest='subpolicy_clamp_value',type=float,default=-5)
	parser.add_argument('--latent_clamp_value',dest='latent_clamp_value',type=float,default=-5)
	parser.add_argument('--min_variance_bias',dest='min_variance_bias',type=float,default=0.01)
	parser.add_argument('--normalization',dest='normalization',type=str,default='None')

	parser.add_argument('--likelihood_penalty',dest='likelihood_penalty',type=int,default=10)
	parser.add_argument('--subpolicy_ratio',dest='subpolicy_ratio',type=float,default=0.01)
	parser.add_argument('--latentpolicy_ratio',dest='latentpolicy_ratio',type=float,default=0.1)
	parser.add_argument('--temporal_latentpolicy_ratio',dest='temporal_latentpolicy_ratio',type=float,default=0.)
	parser.add_argument('--latent_loss_weight',dest='latent_loss_weight',type=float,default=0.1)
	parser.add_argument('--kl_weight',dest='kl_weight',type=float,default=0.01)
	parser.add_argument('--var_loss_weight',dest='var_loss_weight',type=float,default=1.)
	parser.add_argument('--prior_weight',dest='prior_weight',type=float,default=0.00001)

	parser.add_argument('--epsilon_from',dest='epsilon_from',type=float,default=0.3)
	parser.add_argument('--epsilon_to',dest='epsilon_to',type=float,default=0.05)
	parser.add_argument('--epsilon_over',dest='epsilon_over',type=int,default=30)

	return parser.parse_args()

def main(args):

	args = parse_arguments()
	master = Master(args)
	master.run()

if __name__=='__main__':
	main(sys.argv)

