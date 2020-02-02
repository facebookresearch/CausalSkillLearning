#!/usr/bin/env python
from headers import *
import DataLoaders, MIME_DataLoader, Roboturk_DataLoader, Mocap_DataLoader
from PolicyManagers import PolicyManager_Joint, PolicyManager_Pretrain, PolicyManager_DownstreamRL, PolicyManager_DMPBaselines, PolicyManager_BaselineRL, PolicyManager_Imitation
import TestClass

class Master():

	def __init__(self, arguments):
		self.args = arguments 

		# Define Data Loader. 
		if self.args.data=='DeterGoal':
			self.dataset = DataLoaders.DeterministicGoalDirectedDataset(self.args.datadir)			
		elif self.args.data=='MIME':
			self.dataset = MIME_DataLoader.MIME_NewDataset()
		elif self.args.data=='Roboturk':		
			self.dataset = Roboturk_DataLoader.Roboturk_NewSegmentedDataset(self.args)
		elif self.args.data=='OrigRoboturk':
			self.dataset = Roboturk_DataLoader.Roboturk_Dataset(self.args)
		elif self.args.data=='FullRoboturk':
			self.dataset = Roboturk_DataLoader.Roboturk_FullDataset(self.args)
		elif self.args.data=='Mocap':
			self.dataset = Mocap_DataLoader.Mocap_Dataset(self.args)

		# Now define policy manager.
		if self.args.setting=='learntsub':
			self.policy_manager = PolicyManager_Joint(self.args.number_policies, self.dataset, self.args)
		elif self.args.setting=='pretrain_sub':
			self.policy_manager = PolicyManager_Pretrain(self.args.number_policies, self.dataset, self.args)
		elif self.args.setting=='baselineRL':
			self.policy_manager = PolicyManager_BaselineRL(args=self.args)
		elif self.args.setting=='downstreamRL':
			self.policy_manager = PolicyManager_DownstreamRL(args=self.args)
		elif self.args.setting=='DMP':			
			self.policy_manager = PolicyManager_DMPBaselines(self.args.number_policies, self.dataset, self.args)
		elif self.args.setting=='imitation':
			self.policy_manager = PolicyManager_Imitation(self.args.number_policies, self.dataset, self.args)

		if self.args.debug:
			embed()
			
		# Create networks and training operations. 
		self.policy_manager.setup()

	def run(self):
		if self.args.setting=='pretrain_sub' or self.args.setting=='pretrain_prior' or self.args.setting=='imitation' or self.args.setting=='baselineRL' or self.args.setting=='downstreamRL':
			if self.args.train:
				if self.args.model:
					self.policy_manager.train(self.args.model)
				else:
					self.policy_manager.train()
			else:
				if self.args.setting=='pretrain_prior':
					self.policy_manager.train(self.args.model)
				else:
					self.policy_manager.evaluate(model=self.args.model)		
				
		elif self.args.setting=='learntsub':
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

		# elif self.args.setting=='baselineRL' or self.args.setting=='downstreamRL':
		# 	if self.args.train:
		# 		if self.args.model:
		# 			self.policy_manager.train(self.args.model)
		# 		else:
		# 			self.policy_manager.train()

		elif self.args.setting=='DMP':
			self.policy_manager.evaluate_across_testset()

	def test(self):
		if self.args.test_code:
			loader = TestClass.TestLoaderWithKwargs()
			suite = loader.loadTestsFromTestCase(TestClass.MetaTestClass, policy_manager=self.policy_manager)
			unittest.TextTestRunner().run(suite)

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
	parser.add_argument('--test_code',dest='test_code',type=int,default=0)
	parser.add_argument('--model',dest='model',type=str)
	parser.add_argument('--logdir',dest='logdir',type=str,default='Experiment_Logs/')
	parser.add_argument('--epochs',dest='epochs',type=int,default=500) # Number of epochs to train for. Reduce for Mocap.

	# Training setting. 
	parser.add_argument('--discrete_z',dest='discrete_z',type=int,default=0)
	# parser.add_argument('--transformer',dest='transformer',type=int,default=0)	
	parser.add_argument('--z_dimensions',dest='z_dimensions',type=int,default=64)
	parser.add_argument('--number_layers',dest='number_layers',type=int,default=5)
	parser.add_argument('--hidden_size',dest='hidden_size',type=int,default=64)
	parser.add_argument('--environment',dest='environment',type=str,default='SawyerLift') # Defines robosuite environment for RL.
	
	# Data parameters. 
	parser.add_argument('--traj_segments',dest='traj_segments',type=int,default=1) # Defines whether to use trajectory segments for pretraining or entire trajectories. Useful for baseline implementation.
	parser.add_argument('--gripper',dest='gripper',type=int,default=1) # Whether to use gripper training in roboturk.
	parser.add_argument('--ds_freq',dest='ds_freq',type=int,default=1) # Additional downsample frequency.
	parser.add_argument('--condition_size',dest='condition_size',type=int,default=4)
	parser.add_argument('--smoothen', dest='smoothen',type=int,default=0) # Whether to smoothen the original dataset. 
	parser.add_argument('--smoothing_kernel_bandwidth', dest='smoothing_kernel_bandwidth',type=float,default=3.5) # The smoothing bandwidth that is applied to data loader trajectories. 

	parser.add_argument('--new_gradient',dest='new_gradient',type=int,default=1)
	parser.add_argument('--b_prior',dest='b_prior',type=int,default=1)
	parser.add_argument('--reparam',dest='reparam',type=int,default=1)	
	parser.add_argument('--number_policies',dest='number_policies',type=int,default=4)
	parser.add_argument('--fix_subpolicy',dest='fix_subpolicy',type=int,default=1)
	parser.add_argument('--train_only_policy',dest='train_only_policy',type=int,default=0) # Train only the policy network and use a pretrained encoder. This is weird but whatever. 
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
	
	parser.add_argument('--pretrain_bias_sampling',type=float,default=0.) # Defines percentage of trajectory within which to sample trajectory segments for pretraining.
	parser.add_argument('--pretrain_bias_sampling_prob',type=float,default=0.)
	parser.add_argument('--action_scale_factor',type=float,default=1)

	parser.add_argument('--z_exploration_bias',dest='z_exploration_bias',type=float,default=0.)
	parser.add_argument('--b_exploration_bias',dest='b_exploration_bias',type=float,default=0.)
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

	# Exploration and learning rate parameters. 
	parser.add_argument('--epsilon_from',dest='epsilon_from',type=float,default=0.3)
	parser.add_argument('--epsilon_to',dest='epsilon_to',type=float,default=0.05)
	parser.add_argument('--epsilon_over',dest='epsilon_over',type=int,default=30)
	parser.add_argument('--learning_rate',dest='learning_rate',type=float,default=1e-4)

	# Baseline parameters. 
	parser.add_argument('--baseline_kernels',dest='baseline_kernels',type=int,default=15)
	parser.add_argument('--baseline_window',dest='baseline_window',type=int,default=15)
	parser.add_argument('--baseline_kernel_bandwidth',dest='baseline_kernel_bandwidth',type=float,default=3.5)

	# Reinforcement Learning parameters. 
	parser.add_argument('--TD',dest='TD',type=int,default=0) # Whether or not to use Temporal difference while training the critic network.
	parser.add_argument('--OU',dest='OU',type=int,default=1) # Whether or not to use the Ornstein Uhlenbeck noise process while training.
	parser.add_argument('--OU_max_sigma',dest='OU_max_sigma',type=float,default=0.2) # Max Sigma value of the Ornstein Uhlenbeck noise process.
	parser.add_argument('--OU_min_sigma',dest='OU_min_sigma',type=float,default=0.2) # Min Sigma value of the Ornstein Uhlenbeck noise process.
	parser.add_argument('--MLP_policy',dest='MLP_policy',type=int,default=0) # Whether or not to use MLP policy.
	parser.add_argument('--mean_nonlinearity',dest='mean_nonlinearity',type=int,default=0) # Whether or not to use Tanh activation.
	parser.add_argument('--burn_in_eps',dest='burn_in_eps',type=int,default=500) # How many epsiodes to burn in.
	parser.add_argument('--random_memory_burn_in',dest='random_memory_burn_in',type=int,default=1) # Whether to burn in episodes into memory randomly or not.
	parser.add_argument('--shaped_reward',dest='shaped_reward',type=int,default=0) # Whether or not to use shaped rewards.
	parser.add_argument('--memory_size',dest='memory_size',type=int,default=2000) # Size of replay memory. 2000 is okay, but is still kind of short sighted. 


	return parser.parse_args()

def main(args):

	args = parse_arguments()
	master = Master(args)

	if args.test_code:
		master.test()
	else:
		master.run()

if __name__=='__main__':
	main(sys.argv)

