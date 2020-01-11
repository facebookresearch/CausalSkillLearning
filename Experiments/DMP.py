#!/usr/bin/env python2
from .headers import *

class DMP():
	
	# def __init__(self, time_steps=100, num_ker=25, dimensions=3, kernel_bandwidth=None, alphaz=None, time_basis=False):
	def __init__(self, time_steps=40, num_ker=15, dimensions=7, kernel_bandwidth=3.5, alphaz=5., time_basis=True):
		# DMP(dimensions=7,time_steps=40,num_ker=15,kernel_bandwidth=3.5,alphaz=5.,time_basis=True)

		# self.alphaz = 25.0
		if alphaz is not None:
			self.alphaz = alphaz
		else:
			self.alphaz = 10.
		self.betaz = self.alphaz/4
		self.alpha = self.alphaz/3
		
		self.time_steps = time_steps
		self.tau = self.time_steps
		# self.tau = 1.
		self.use_time_basis = time_basis

		self.dimensions = dimensions
		# self.number_kernels = max(500,self.time_steps)
		self.number_kernels = num_ker
		if kernel_bandwidth is not None:
			self.kernel_bandwidth = kernel_bandwidth
		else:
			self.kernel_bandwidth = self.calculate_good_sigma(self.time_steps, self.number_kernels)
		self.epsilon = 0.001
		self.setup()
		
	def setup(self):

		self.gaussian_kernels = np.zeros((self.number_kernels,2))

		self.weights = np.zeros((self.number_kernels, self.dimensions))

		self.demo_pos = np.zeros((self.time_steps, self.dimensions))
		self.demo_vel = np.zeros((self.time_steps, self.dimensions))
		self.demo_acc = np.zeros((self.time_steps, self.dimensions))

		self.target_forces = np.zeros((self.time_steps, self.dimensions))        
		self.phi = np.zeros((self.number_kernels, self.time_steps, self.time_steps))
		self.eta = np.zeros((self.time_steps, self.dimensions))
		self.vector_phase = np.zeros(self.time_steps)	
        
		# Defining Rollout variables.
		self.rollout_time = self.time_steps
		self.dt = 1./self.rollout_time
		self.pos_roll = np.zeros((self.rollout_time,self.dimensions))
		self.vel_roll = np.zeros((self.rollout_time,self.dimensions))
		self.acc_roll = np.zeros((self.rollout_time,self.dimensions))
		self.force_roll = np.zeros((self.rollout_time,self.dimensions))        
		self.goal = np.zeros(self.dimensions)
		self.start = np.zeros(self.dimensions)        

	def calculate_good_sigma(self, time, number_kernels, threshold=0.15):
		return time/(2*(number_kernels-1)*(np.sqrt(-np.log(threshold))))

	def load_trajectory(self,pos,vel=None,acc=None):

		self.demo_pos = np.zeros((self.time_steps, self.dimensions))
		self.demo_vel = np.zeros((self.time_steps, self.dimensions))
		self.demo_acc = np.zeros((self.time_steps, self.dimensions))

		if vel is not None and acc is not None: 
			self.demo_pos = copy.deepcopy(pos)
			self.demo_vel = copy.deepcopy(vel)
			self.demo_acc = copy.deepcopy(acc)
		else: 
			self.smooth_interpolate(pos)

	def smooth_interpolate(self, pos):
		# Filter the posiiton input by Gaussian smoothing. 
		smooth_pos = gaussian_filter1d(pos,3.5,axis=0,mode='nearest')		

		time_range = np.linspace(0, pos.shape[0]-1, pos.shape[0])
		new_time_range = np.linspace(0,pos.shape[0]-1,self.time_steps+2)

		self.interpolated_pos = np.zeros((self.time_steps+2,self.dimensions))
		interpolating_objects = []

		for i in range(self.dimensions):
			interpolating_objects.append(interp1d(time_range,pos[:,i],kind='linear'))
			self.interpolated_pos[:,i] = interpolating_objects[i](new_time_range)

		self.demo_vel = np.diff(self.interpolated_pos,axis=0)[:self.time_steps]
		self.demo_acc = np.diff(self.interpolated_pos,axis=0,n=2)[:self.time_steps]
		self.demo_pos = self.interpolated_pos[:self.time_steps]

	def initialize_variables(self):	
		self.weights = np.zeros((self.number_kernels, self.dimensions))
		self.target_forces = np.zeros((self.time_steps, self.dimensions))
		self.phi = np.zeros((self.number_kernels, self.time_steps, self.time_steps))
		self.eta = np.zeros((self.time_steps, self.dimensions))

		self.kernel_centers = np.linspace(0,self.time_steps,self.number_kernels)

		self.vector_phase = self.calc_vector_phase(self.kernel_centers)
		self.gaussian_kernels[:,0] = self.vector_phase

		# Different kernel parameters that have worked before, giving different behavior. 		
		# # dummy = (np.diff(self.gaussian_kernels[:,0]*0.55))**2        		
		# # dummy = (np.diff(self.gaussian_kernels[:,0]*2))**2
		# # dummy = (np.diff(self.gaussian_kernels[:,0]))**2        						

		dummy = (np.diff(self.gaussian_kernels[:,0]*self.kernel_bandwidth))**2
		self.gaussian_kernels[:,1] = 1. / np.append(dummy,dummy[-1])
		
		# self.gaussian_kernels[:,1] = self.number_kernels/self.gaussian_kernels[:,0]

	def calc_phase(self,time):
		return np.exp(-self.alpha*float(time)/self.tau)

	def calc_vector_phase(self,time):
		return np.exp(-self.alpha*time.astype(float)/self.tau)

	def basis(self,index,time):
		return np.exp(-(self.gaussian_kernels[index,1])*((self.calc_phase(time)-self.gaussian_kernels[index,0])**2))

	def time_basis(self, index, time):
		# return np.exp(-(self.gaussian_kernels[index,1])*((time-self.kernel_centers[index])**2))
		# return np.exp(-(time-self.kernel_centers[index])**2)
		return np.exp(-((time-self.kernel_centers[index])**2)/(self.kernel_bandwidth))

	def vector_basis(self, index, time_range):
		return np.exp(-(self.gaussian_kernels[index,1])*((self.calc_vector_phase(time_range)-self.gaussian_kernels[index,0])**2))

	def update_target_force_itau(self):
		self.target_forces = (self.tau**2)*self.demo_acc - self.alphaz*(self.betaz*(self.demo_pos[self.time_steps-1]-self.demo_pos)-self.tau*self.demo_vel)

	def update_target_force_dtau(self):
		self.target_forces = self.demo_acc/(self.tau**2) - self.alphaz*(self.betaz*(self.demo_pos[self.time_steps-1]-self.demo_pos)-self.demo_vel/self.tau)    

	def update_target_force(self):
		self.target_forces = self.demo_acc - self.alphaz*(self.betaz*(self.demo_pos[self.time_steps-1]-self.demo_pos)-self.demo_vel)

	def update_phi(self):		
		for i in range(self.number_kernels):
			for t in range(self.time_steps):
				if self.use_time_basis:
					self.phi[i,t,t] = self.time_basis(i,t)
				else:
					self.phi[i,t,t] = self.basis(i,t)
                
	def update_eta(self):        
		t_range = np.linspace(0,self.time_steps,self.time_steps)        
		vector_phase = self.calc_vector_phase(t_range)        

		for k in range(self.dimensions):
			self.eta[:,k] = vector_phase*(self.demo_pos[self.time_steps-1,k]-self.demo_pos[0,k])

	def learn_DMP(self, pos, forces="i"):
		self.setup()
		self.load_trajectory(pos)
		self.initialize_variables()
		self.learn_weights(forces=forces)

	def learn_weights(self, forces="i"):

		if forces=="i":
			self.update_target_force_itau()    
		elif forces=="d":
			self.update_target_force_dtau() 
		elif forces=="n":
			self.update_target_force() 
		self.update_phi()
		self.update_eta()

		for j in range(self.dimensions):
			for i in range(self.number_kernels):
				self.weights[i,j] = np.dot(self.eta[:,j],np.dot(self.phi[i],self.target_forces[:,j]))
				self.weights[i,j] /= np.dot(self.eta[:,j],np.dot(self.phi[i],self.eta[:,j])) + self.epsilon 	

	def initialize_rollout(self,start,goal,init_vel):

		self.pos_roll = np.zeros((self.rollout_time,self.dimensions))
		self.vel_roll = np.zeros((self.rollout_time,self.dimensions))
		self.acc_roll = np.zeros((self.rollout_time,self.dimensions))

		self.tau = self.rollout_time		
		self.pos_roll[0] = copy.deepcopy(start)
		self.vel_roll[0] = copy.deepcopy(init_vel)
		self.goal = goal
		self.start = start
		self.dt = self.tau/self.rollout_time   		
		# print(self.dt,self.tau,self.rollout_time)

	def calc_rollout_force(self, roll_time):
		den = 0        		
		time = copy.deepcopy(roll_time)
		for i in range(self.number_kernels):
			
			if self.use_time_basis:
				self.force_roll[roll_time] += self.time_basis(i,time)*self.weights[i]
				den += self.time_basis(i,time)
			else:
				self.force_roll[roll_time] += self.basis(i,time)*self.weights[i]
				den += self.basis(i,time)
			
		self.force_roll[roll_time] *= (self.goal-self.start)*self.calc_phase(time)/den
        
	def calc_rollout_acceleration(self,time):        
		self.acc_roll[time] = (1./self.tau**2)*(self.alphaz * (self.betaz * (self.goal - self.pos_roll[time]) - self.tau*self.vel_roll[time]) + self.force_roll[time])
        
	def calc_rollout_vel(self,time):		
		self.vel_roll[time] = self.vel_roll[time-1] + self.acc_roll[time-1]*self.dt

	def calc_rollout_pos(self,time):
		self.pos_roll[time] = self.pos_roll[time-1] + self.vel_roll[time-1]*self.dt

	def rollout(self,start,goal,init_vel):
		self.initialize_rollout(start,goal,init_vel)
		self.calc_rollout_force(0)
		self.calc_rollout_acceleration(0)
		for i in range(1,self.rollout_time):        
			self.calc_rollout_force(i)		
			self.calc_rollout_vel(i)
			self.calc_rollout_pos(i)   
			self.calc_rollout_acceleration(i)
		return self.pos_roll

	def load_weights(self, weight):
		self.weights = copy.deepcopy(weight)	

def main(args):    

	pos = np.load(str(sys.argv[1]))[:,:3]
	vel = np.load(str(sys.argv[2]))[:,:3]
	acc = np.load(str(sys.argv[3]))[:,:3]

	rolltime = 500
	dmp = DMP(rolltime)

	dmp.load_trajectory(pos)
	dmp.initialize_variables()
	dmp.learn_DMP()
		
	start = np.zeros(dmp.dimensions)	
	goal = np.ones(dmp.dimensions)
	norm_vector = pos[-1]-pos[0]
	init_vel = np.divide(vel[0],norm_vector)	

	dmp.rollout(start, goal, init_vel)	
	dmp.save_rollout()

