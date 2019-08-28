#!/usr/bin/env python
from headers import *
import GridWorld_DataLoader, SmallMaps_DataLoader, ContinuousToyDataLoader, ContinuousDirectedToyDataLoader
import PolicyManager_GTSubpolicy, PolicyManager_GTSubpolicy_ActionLikelihood, PolicyManager_LearntDiscreteSubpolicy
import Pretrain_Subpolicy

class Master():

	def __init__(self, arguments):
		self.args = arguments 

		# Define Data Loader. 
		if self.args.data=='Bigmaps':
			self.dataset = GridWorld_DataLoader.GridWorldDataset(self.args.datadir)
		elif self.args.data=='Smallmaps':
			self.dataset = SmallMaps_DataLoader.GridWorldDataset(self.args.datadir)
		elif self.args.data=='ToyData':
			self.dataset = ToyDataLoader.ToyDataset(self.args.datadir)
		elif self.args.data=='Continuous':
			self.dataset = ContinuousToyDataLoader.ToyDataset(self.args.datadir)
		elif self.args.data=='ContinuousDir':
			self.dataset = ContinuousDirectedToyDataLoader.ToyDataset(self.args.datadir)

		self.number_policies = 4
		if self.args.setting=='just_actions':
			self.policy_manager = PolicyManager_GTSubpolicy_ActionLikelihood.PolicyManager(self.number_policies, self.dataset, self.args)
		elif self.args.setting=='gtsub':
			self.policy_manager = PolicyManager_GTSubpolicy.PolicyManager(self.number_policies, self.dataset, self.args)
		elif self.args.setting=='learntsub':
			self.policy_manager = PolicyManager_LearntDiscreteSubpolicy.PolicyManager(self.number_policies, self.dataset, self.args)
		elif self.args.setting=='pretrain_sub':
			self.policy_manager = Pretrain_Subpolicy.PolicyManager(self.number_policies, self.dataset, self.args)
			
		# Create networks and training operations. 
		self.policy_manager.setup()

	def run(self):
		if self.args.setting=='pretrain_sub':			
			if self.args.train:
				self.policy_manager.train()
			else:
				self.policy_manager.evaluate(self.args.model)
		else:
			if self.args.train:
				if self.args.model:
					self.policy_manager.train(self.args.model)
				else:
					if self.args.subpolicy_model:
						self.policy_manager.load_all_models(self.args.subpolicy_model, just_subpolicy=True)
					self.policy_manager.train()
			else:
				self.policy_manager.evaluate(self.args.model)

def parse_arguments():
	parser = argparse.ArgumentParser(description='Learning Skills from Demonstrations')
	parser.add_argument('--datadir', dest='datadir',type=str,default='../../DataGenerator/ContData/')
	parser.add_argument('--train',dest='train',type=int,default=0)
	parser.add_argument('--data',dest='data',type=str,default='Continuous')
	parser.add_argument('--model',dest='model',type=str)
	parser.add_argument('--subpolicy_model',dest='subpolicy_model',type=str)
	parser.add_argument('--notes',dest='notes',type=str)
	parser.add_argument('--name',dest='name',type=str,default=None)
	parser.add_argument('--entropy',dest='entropy',type=int,default=0)
	parser.add_argument('--var_entropy',dest='var_entropy',type=int,default=0)
	parser.add_argument('--ent_weight',dest='ent_weight',type=float,default=1.)
	parser.add_argument('--var_ent_weight',dest='var_ent_weight',type=float,default=2.)
	parser.add_argument('--z_ex_bias',dest='z_exploration_bias',type=float,default=0.)
	parser.add_argument('--b_ex_bias',dest='b_exploration_bias',type=float,default=0.)
	parser.add_argument('--setting',dest='setting',type=str,default='gtsub')
	parser.add_argument('--display_freq',dest='display_freq',type=int,default=1000)
	parser.add_argument('--expert',dest='expert',type=int,default=0)
	parser.add_argument('--logdir',dest='logdir',type=str,default='Experiment_Logs/')
	parser.add_argument('--traj_length',dest='traj_length',type=int,default=10)
	parser.add_argument('--likelihood_penalty',dest='likelihood_penalty',type=int,default=10)
	parser.add_argument('--subpolicy_ratio',dest='subpolicy_ratio',type=float,default=0.01)
	parser.add_argument('--epsilon_from',dest='epsilon_from',type=float,default=0.5)
	parser.add_argument('--epsilon_to',dest='epsilon_to',type=float,default=0.1)
	parser.add_argument('--epsilon_over',dest='epsilon_over',type=int,default=10)

	return parser.parse_args()

def main(args):

	args = parse_arguments()
	master = Master(args)
	master.run()

if __name__=='__main__':
	main(sys.argv)