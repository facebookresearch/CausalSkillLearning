#!/usr/bin/env python
from headers import *

class MetaTestClass(unittest.TestCase):

	def __init__(self, args, policy_manager, dataset):		
		super(MetaTestClass, self).__init__()
		self.args = args
		self.policy_manager = policy_manager
		self.dataset = dataset

	# def setup(self, args, policy_manager, dataset):
	# 	self.args = args
	# 	self.policy_manager = policy_manager
	# 	self.dataset = dataset

		# self.subpolicy = subpolicy
		# if latent_policy is not None: 
		# 	self.latent_policy = latent_policy
		# if variational_policy is not None: 
		# 	self.variational_policy = variational_policy
	
	# def runTest(self):
	# 	pass

	def test_dataloader(self):

		if self.args.data=='Roboturk':
			self.check_Roboturkdataloader()
		if self.args.data=='MIME':
			self.check_MIMEdataloader()

	def check_MIMEdataloader(self):

		self.dataset = MIME_NewDataset()

		# Check the first index of the dataset.
		data_element = self.dataset[0]

		validity = data_element['is_valid']==1
		check_demo_data = (data_element['demo']==np.load("Test_Data/MIME_Dataloader_DE.npy")).all()

		self.assertTrue(validity and check_demo_data)

	def check_Roboturkdataloader(self):

		# Check the first index of the dataset.
		data_element = self.dataset[0]

		validity = data_element['is_valid']
		check_demo_data = (data_element['demo']==np.load("Test_Data/Roboturk_Dataloader_DE.npy")).all()

		self.assertTrue(validity and check_demo_data)

	def test_variational_policy(self):

		# Assume the variational policy is an instance of ContinuousVariationalPolicyNetwork_BPrior class.
		pass

	def test_subpolicy(self):

		# Assume the subpolicy is an instance of ContinuousPolicyNetwork class.

		inputs = torch.ones((15,self.policy_manager.policy_network.input_size)).cuda().float()
		actions = np.ones((15,self.policy_manager.policy_network.output_size)).cuda().float()

		expected_outputs = np.load("{0}_Subpolicy_Res.npy".format(self.args.data),allow_pickle=True)
		pred_outputs = self.policy_manager.policy_network.forward(inputs, actions)

		error = (((expected_outputs[0]-pred_outputs[0])**2).mean()).cpu().numpy()
		threshold = 0.01

		self.assertTrue(error < threshold)

	def test_latent_policy(self):

		# Assume the latent policy is a ContinuousLatentPolicyNetwork class.
		pass

	def test_encoder_policy(self):

		# Assume is instance of ContinuousEncoderNetwork class.
		pass