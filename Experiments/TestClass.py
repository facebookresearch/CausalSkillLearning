# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from headers import *

class TestLoaderWithKwargs(unittest.TestLoader):
    """A test loader which allows to parse keyword arguments to the
       test case class."""
    # def loadTestsFromTestCase(self, testCaseClass, **kwargs):
    def loadTestsFromTestCase(self, testCaseClass, policy_manager):
        """Return a suite of all tests cases contained in 
           testCaseClass."""
        if issubclass(testCaseClass, unittest.suite.TestSuite):
            raise TypeError("Test cases should not be derived from "\
                            "TestSuite. Maybe you meant to derive from"\
                            " TestCase?")
        testCaseNames = self.getTestCaseNames(testCaseClass)
        if not testCaseNames and hasattr(testCaseClass, 'runTest'):
            testCaseNames = ['runTest']

        # Modification here: parse keyword arguments to testCaseClass.
        test_cases = []

        # embed()
        for test_case_name in testCaseNames:
            # test_cases.append(testCaseClass(policy_manager))
            test_cases.append(testCaseClass(test_case_name, policy_manager))
        loaded_suite = self.suiteClass(test_cases)

        return loaded_suite 

class MetaTestClass(unittest.TestCase):

	def __init__(self, test_name, policy_manager):		
		super(MetaTestClass, self).__init__(test_name)
		
		self.policy_manager = policy_manager
		self.args = self.policy_manager.args
		self.dataset = self.policy_manager.dataset

	def test_dataloader(self):

		if self.args.data=='Roboturk':
			self.check_Roboturkdataloader()
		if self.args.data=='MIME':
			self.check_MIMEdataloader()

	def check_MIMEdataloader(self):

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

		if self.args.setting=='learntsub':
			# Assume the variational policy is an instance of ContinuousVariationalPolicyNetwork_BPrior class.		
			inputs = torch.ones((40,self.policy_manager.variational_policy.input_size)).cuda().float()

			expected_outputs = np.load("Test_Data/{0}_Varpolicy_Res.npy".format(self.args.data),allow_pickle=True)
			pred_outputs = self.policy_manager.variational_policy.forward(inputs, epsilon=0.)
			error = (((expected_outputs[0]-pred_outputs[0])**2).mean()).detach().cpu().numpy()

			threshold = 0.01

			self.assertTrue(error < threshold)
		else:
			pass

	def test_subpolicy(self):

		# Assume the subpolicy is an instance of ContinuousPolicyNetwork class.
		inputs = torch.ones((15,self.policy_manager.policy_network.input_size)).cuda().float()
		actions = np.ones((15,self.policy_manager.policy_network.output_size))

		expected_outputs = np.load("Test_Data/{0}_Subpolicy_Res.npy".format(self.args.data),allow_pickle=True)
		pred_outputs = self.policy_manager.policy_network.forward(inputs, actions)

		error = (((expected_outputs[0]-pred_outputs[0])**2).mean()).detach().cpu().numpy()

		threshold = 0.01

		self.assertTrue(error < threshold)

	def test_latent_policy(self):

		# Assume the latent policy is a ContinuousLatentPolicyNetwork class.
		pass

	def test_encoder_policy(self):

		# Assume is instance of ContinuousEncoderNetwork class.
		pass