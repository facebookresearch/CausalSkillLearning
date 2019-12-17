#!/usr/bin/env python

# **Here are the modifications we make to the Machine Translation Transformer Model:**
# 1. Replacing the Embeddings used with "vocabulary" with (shared) linear Embedding Layers.
# 2. Replacing input and output vocabulary (sizes) with state_size. (Remember, this also affects the Generator). 
# 
# 3. We must change the output heads of the decoder (Generator?) to regress to continuous values and also predict stop probabilities, rather than predict a softmax over the vocabulary. 
# 4. Change loss function from the supervised cross-entropy loss to the L2 regression loss. 

import numpy as np, math, copy, time, pdb
import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from IPython import embed
from memory_profiler import profile
import gc
from absl import flags

# flags.DEFINE_boolean('presoftmax_bias', True, 'Whether or not to apply continuing_bias before stop probability softmax.')
# flags.DEFINE_float('b_probability_factor', 0.01, 'Factor to multiply preprobabilities with before adding bias.')

class TransformerBaseClass(nn.Module):
	
	# Define a transformer that takes in an Encoder-Decoder and puts a "generator" (output layer) on top of it.       
	# def __init__(self, opts, number_layers=6, attention_heads=8, dropout=0.1, dummy_inputs=False, maximum_skills=5):
	def __init__(self, input_size, hidden_size, z_dimensionality, args, number_layers=6, attention_heads=8, dropout=0.1):
		
		super(TransformerBaseClass, self).__init__()
		"Helper: Construct a model from hyperparameters."

		self.args = args
		d_model = hidden_size
		d_ff = d_model*2
		state_size = input_size
		output_size = z_dimensionality
		self.z_dimensionality = z_dimensionality
		self.output_size = output_size
		
		# self.opts = opts
		# d_model = 64
		# d_ff = self.opts.nh*2
		# state_size = self.opts.n_state
		# output_size = self.opts.nz

		h = attention_heads
		N = number_layers
		self.dummy_inputs = False
		# self.maximum_length = maximum_skills

		c = copy.deepcopy
		attn = MultiHeadedAttention(h, d_model)
		ff = PositionwiseFeedForward(d_model, d_ff, dropout)
		
		# Create positional encoding.
		position = PositionalEncoding(d_model, dropout)
			
		# Creating instance of embedding layer.
		# embedding_layer = EmbeddingLayer(d_model, state_size)       
		source_embedding_layer = EmbeddingLayer(state_size, d_model)  
		target_embedding_layer = EmbeddingLayer(output_size, d_model)

		# Set continuing bias.
		# self.continuing_bias = self.opts.lpred_p_bias

		# Create the Transformer. 
		self.encoder_decoder = EncoderDecoder(
							Encoder( EncoderLayer(d_model, c(attn), c(ff), dropout), N),
							Decoder( DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
							nn.Sequential( c(source_embedding_layer), c(position)),
							nn.Sequential( c(target_embedding_layer), c(position)))
		
		# This was important from their code. 
		# Initialize parameters with Glorot / fan_avg.
		for p in self.encoder_decoder.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)

		# self.generator = Generator(d_model, output_size, variable_timesteps=self.opts.variable_nseg, presoftmax_bias=opts.presoftmax_bias)

	# Modifying this forward function for Skill Net!
	# 1) Instead of feeding in the target as input to the Transformer straight out, we encode the source,
	#    and by feeding in a dummy target, we decode to retrieve a prediction of Z and stop_probability (if variable nseg).
	# 2) We have two options with regard to feeding in the target:
	#       a) We append the predicted Z to the dummy target, and rebatch and decode again. 
	#       b) We feed just zeros as the dummy target. 
	# 3) Depending on if we have fixed of variable number of segments, we either: 
	#       a) Keep decoding till a fixed number of steps in the target. 
	#       b) Sample stop variable from stop_probability predicted, and continue is stop is False. 
	# * Special note: If we are using fixed number of timesteps and zeros as dummy input, this may be done
	#   by setting the target beforehand. 


class TransformerVariationalNet(TransformerBaseClass):

	def __init__(self):

		pass

class TransformerEncoder(TransformerBaseClass):
	# Class to select one z from a trajectory segment. 
	def __init__(self, input_size, hidden_size, z_dimensionality, args, number_layers=6, attention_heads=8, dropout=0.1):

		super(TransformerEncoder, self).__init__(input_size, hidden_size, z_dimensionality, args, number_layers, attention_heads, dropout)
		
		d_model = hidden_size
		# Define output layers for the LSTM, and activations for this output layer. 
		self.mean_output_layer = torch.nn.Linear(d_model,self.z_dimensionality)
		self.variances_output_layer = torch.nn.Linear(d_model, self.z_dimensionality)
		self.variance_activation_layer = torch.nn.Softplus()
		self.variance_factor = 0.01

	def forward(self, source, epsilon=0.001, z_sample_to_evaluate=None):

		datapoint = MultiDimensionalBatch(source)
		
		# Encode the source sequence into "memory".
		memory = self.encoder_decoder.encode(datapoint.source, datapoint.source_mask)

		# Initialize dummy target. 
		target = torch.zeros((1,self.z_dimensionality)).float().cuda()

		# Rebatch and create target_masks.
		datapoint = MultiDimensionalBatch(source, target)

		# Decode memory with target.                   
		decoded_output = self.encoder_decoder.decode(memory, datapoint.source_mask, datapoint.target, datapoint.target_mask)

		# Here, we do NOT need to select the last output of decoded_output, because the target provided itself was just ONE Z, meaning our outputs are also just ONE Z.
		# Compute mean and variance.
		mean_outputs = self.mean_output_layer(decoded_output)
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(decoded_output)))+ epsilon

		# Construct distribution. 
		dist = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))

		# Reparametrization based z construction. 
		noise = torch.randn_like(variance_outputs)
		latent_z = mean_outputs + variance_outputs*noise

		# Calculate entropy for training.
		entropy = dist.entropy()
		# Also retrieve log probability of the same.
		logprobability = dist.log_prob(latent_z)

		# Set standard distribution for KL. 
		standard_distribution = torch.distributions.MultivariateNormal(torch.zeros((self.output_size)).cuda(),torch.eye((self.output_size)).cuda())
		# Compute KL.
		kl_divergence = torch.distributions.kl_divergence(dist, standard_distribution)

		if self.args.debug:
			print("###############################")
			print("Embedding in Encoder Network.")
			embed()

		if z_sample_to_evaluate is None:
			return latent_z, logprobability, entropy, kl_divergence
		else:
			logprobability = dist.log_prob(z_sample_to_evaluate)
			return logprobability

# class TransformerFixedNSeg(TransformerBaseClass):

# 	def __init__(self, opts, number_layers=6, attention_heads=8, dropout=0.1, dummy_inputs=False, maximum_skills=5):

# 		# This just calls __init__ of base class.
# 		super(TransformerFixedNSeg, self).__init__(opts, number_layers, attention_heads, dropout, dummy_inputs, maximum_skills)

# 	# @profile
# 	def forward(self, source, target=None, fake_fnseg=False):

# 		# gc.collect()
# 		# * Special case * Implementation of forward if we use fixed number of timesteps and zeros as inputs. 
# 		if self.dummy_inputs:
# 			# Set targets to torch zeros of size number of timesteps and size Z embedding.
# 			target = torch.zeros((self.opts.n_skill_segments,self.opts.nz)).float().cuda()
# 			datapoint = MultiDimensionalBatch(source,target)
# 			enc_dec_output = self.encoder_decoder.forward(datapoint.source, datapoint.target, datapoint.source_mask, datapoint.target_mask)
# 			return self.generator(enc_dec_output) 
		
# 		else:
# 			# Common steps whether or not we are variable number of skills. 
# 			# Create the datapoint.
# 			datapoint = MultiDimensionalBatch(source)
			
# 			# Encode the source sequence into "memory".
# 			memory = self.encoder_decoder.encode(datapoint.source, datapoint.source_mask)

# 			# Initialize dummy target. 
# 			target = torch.zeros((1,self.opts.nz)).float().cuda()

# 			# For fixed number of skills = self.opts.n_skill_segments

# 			# For number of skill segments
# 			for t in range(self.opts.n_skill_segments):

# 				# Rebatch and create target_masks.
# 				datapoint = MultiDimensionalBatch(source, target)

# 				# Decode memory with target.                   
# 				decoded_output = self.encoder_decoder.decode(memory, datapoint.source_mask, datapoint.target, datapoint.target_mask)
# 				latent_z = self.generator.forward(decoded_output)

# 				if len(latent_z.shape)>2:
# 					latent_z = latent_z.squeeze(0)
# 				# Decode actually gives you the entire sequence of Zs. (Which are guaranteed to be the same every time, because deterministic output layer).

# 				# Since Dummy Input can't be true here, concatenate the last Z to target and go again.                    
# 				target = torch.cat([target, latent_z[-1].view((-1,self.opts.nz))],dim=0)

# 			return latent_z.view((-1,self.opts.nz))

# class TransformerVariableNSeg(TransformerBaseClass):

# 	def __init__(self, opts, number_layers=6, attention_heads=8, dropout=0.1, dummy_inputs=False, maximum_skills=5):

# 		# This just calls __init__ of base class.
# 		super(TransformerVariableNSeg, self).__init__(opts, number_layers, attention_heads, dropout, dummy_inputs, maximum_skills)

# 	# @profile
# 	def forward(self, source, target=None, fake_fnseg=False):   
# 		# For variable number of skills. 

# 		# gc.collect()
# 		# Common steps whether or not we are variable number of skills. 
# 		# Create the datapoint.
# 		datapoint = MultiDimensionalBatch(source)
		
# 		# Encode the source sequence into "memory".
# 		memory = self.encoder_decoder.encode(datapoint.source, datapoint.source_mask)

# 		# Initialize dummy target. 
# 		target = torch.zeros((1,self.opts.nz)).float().cuda()

# 		# Initialize stop variables and timestep counter. 
# 		stop = False
# 		t = 0
		
# 		stop_probabilities = None
# 		latent_z_seq = None

# 		# print("######### Starting a new trajectory. ###########")

# 		while not(stop):

# 			# Rebatch and create target_masks.
# 			datapoint = MultiDimensionalBatch(source, target)

# 			# Decode memory with target.
# 			decoded_output = self.encoder_decoder.decode(memory, datapoint.source_mask, datapoint.target, datapoint.target_mask)                                    
# 			# Pass only the last timestep decoded output through the generator. 
# 			latent_z, stop_probability = self.generator.forward(decoded_output[0,-1])

# 			# If we are training a Variable NSeg model with fixed number seg for the first few iterations. 
# 			if fake_fnseg and t>=self.maximum_length:
# 				stop = True
# 				stop_probability = stop_probability*0
# 				stop_probability[1] += 1 
# 			# If Fake FNSEG but not yet at 6 Zs. 
# 			elif fake_fnseg and t<self.maximum_length:
# 				stop = False
# 			else:
# 				epsilon_value = 1e-4
# 				# Instead of assigning new variable make inplace modification. 
# 				stop_probability = stop_probability+epsilon_value                       

# 				# Not adding stop probability here anymore unless opts.presoftmax_bias is false.			
# 				if not(self.opts.presoftmax_bias):
# 					stop_probability[0] = stop_probability[0] + self.continuing_bias
# 				stop_probability = stop_probability/stop_probability.sum()

# 				sample_t = torch.distributions.Categorical(probs=stop_probability).sample()
# 				stop = bool(sample_t == 1)
				
# 			if self.dummy_inputs:
# 				# Concatenate dummy Z to target and go again. 
# 				target = torch.cat([target,torch.zeros((1,self.opts.nz)).cuda().float()],dim=0)
			
# 			else:
# 				# Concatenate the last Z to target and go again. 
# 				try:
# 					target = torch.cat([target, latent_z.view((-1,self.opts.nz))],dim=0)
# 				except:
# 					embed()

# 			# These if blocks are just so that we can use torch tensors and not have to use lists.
# 			if latent_z_seq is None:
# 				latent_z_seq = latent_z.view(-1,self.opts.nz)
# 			else:
# 				latent_z_seq = torch.cat([latent_z_seq,latent_z.view(-1,self.opts.nz)],dim=0)

# 			# These if blocks are just so that we can use torch tensors and not have to use lists
# 			if stop_probabilities is None:
# 				stop_probabilities = stop_probability.view(-1,2)
# 			else:
# 				stop_probabilities = torch.cat([stop_probabilities, stop_probability.view(-1,2)],dim=0)

# 			t+=1 

# 		# print(t)
# 		# return torch.cat(latent_z_seq,dim=0), torch.cat(stop_probabilities,dim=0)
# 		return latent_z_seq, stop_probabilities
# 		# return latent_z.view((-1,self.opts.nz)), stop_probabilities.view((-1,2))

class EncoderDecoder(nn.Module):
	"""
	A standard Encoder-Decoder architecture. Base for this and many 
	other models.
	"""
	def __init__(self, encoder, decoder, src_embed, tgt_embed):
		
		# Class initialization. 
		super(EncoderDecoder, self).__init__()
		
		# Make the encoder and decoder an element of this class. 
		self.encoder = encoder
		self.decoder = decoder
		# Remember, the src_embed and tgt_embed are instances of PyTorch layers that are treated as functions. 
		self.src_embed = src_embed
		self.tgt_embed = tgt_embed    
		
	def forward(self, src, tgt, src_mask, tgt_mask):
		"Take in and process masked src and target sequences."
		return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
	
	def encode(self, src, src_mask):
		return self.encoder(self.src_embed(src), src_mask)
	
	def decode(self, memory, src_mask, tgt, tgt_mask):
		return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

# class DistributionGenerator(nn.Module):
# 	'''"Define standard linear + softmax generation step."'''
	
# 	def __init__(self, d_model, output_size, variable_timesteps=False, presoftmax_bias=True):       
# 		super(Generator, self).__init__()        
		
# 		self.variable_timesteps = variable_timesteps
# 		self.linear_layer = nn.Linear(d_model, output_size)
# 		self.presoftmax_bias = presoftmax_bias
# 		if self.variable_timesteps:
# 			self.stopping_probability_layer = torch.nn.Linear(d_model, 2) 			
# 			self.softmax_layer = torch.nn.Softmax(dim=-1)

# 	def forward(self, x, bias=0.):
# 		if self.variable_timesteps:
# 			preprobs = self.stopping_probability_layer(x)
# 			if self.presoftmax_bias:
# 				preprobs = preprobs*self.opts.b_probability_factor	
# 				preprobs[0] += bias
# 			return self.linear_layer(x), self.softmax_layer(preprobs), 
# 		else:
# 			return self.linear_layer(x)


class Generator(nn.Module):
	'''"Define standard linear + softmax generation step."'''
	
	def __init__(self, d_model, output_size, variable_timesteps=False, presoftmax_bias=True):       
		super(Generator, self).__init__()        
		
		self.variable_timesteps = variable_timesteps
		self.linear_layer = nn.Linear(d_model, output_size)
		self.presoftmax_bias = presoftmax_bias
		if self.variable_timesteps:
			self.stopping_probability_layer = torch.nn.Linear(d_model, 2) 			
			self.softmax_layer = torch.nn.Softmax(dim=-1)

	def forward(self, x, bias=0.):
		if self.variable_timesteps:
			preprobs = self.stopping_probability_layer(x)
			if self.presoftmax_bias:
				preprobs = preprobs*self.opts.b_probability_factor	
				preprobs[0] += bias
			return self.linear_layer(x), self.softmax_layer(preprobs), 
		else:
			return self.linear_layer(x)

def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
	"Core encoder is a stack of N layers"
	def __init__(self, layer, N):
		super(Encoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)
		
	def forward(self, x, mask):
		"Pass the input (and mask) through each layer in turn."
		# print("PRINTING IN ENCODER FORWARD.")
		counter = 0 
		for layer in self.layers:
			# print("Layer: ",counter," with X:",x," and mask:", mask)
			x = layer(x, mask)
			counter += 1 
		return self.norm(x)

class LayerNorm(nn.Module):
	"Construct a layernorm module (See citation for details)."
	def __init__(self, features, eps=1e-6):
		super(LayerNorm, self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps

	def forward(self, x):
		mean = x.mean(-1, keepdim=True)
		std = x.std(-1, keepdim=True)
		return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
	"""
	A residual connection followed by a layer norm.
	Note for code simplicity the norm is first as opposed to last.
	"""
	def __init__(self, size, dropout):
		super(SublayerConnection, self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
	"Encoder is made up of self-attn and feed forward (defined below)"
	def __init__(self, size, self_attn, feed_forward, dropout):
		super(EncoderLayer, self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 2)
		self.size = size

	def forward(self, x, mask):
		"Follow Figure 1 (left) for connections."
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
		return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
	"Generic N layer decoder with masking."
	def __init__(self, layer, N):
		super(Decoder, self).__init__()
		self.layers = clones(layer, N)
		self.norm = LayerNorm(layer.size)
		
	def forward(self, x, memory, src_mask, tgt_mask):
		# print("Running the Decoder.")        
		# print("X shape:",x.shape)
		# print("Memory shape:",memory.shape)
		for layer in self.layers:
			x = layer(x, memory, src_mask, tgt_mask)
		return self.norm(x)

class DecoderLayer(nn.Module):
	"Decoder is made of self-attn, src-attn, and feed forward (defined below)"
	def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
		super(DecoderLayer, self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.src_attn = src_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
	def forward(self, x, memory, src_mask, tgt_mask):
		"Follow Figure 1 (right) for connections."
		m = memory
		# print("Self attention.")
		x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
		# print("Source attention.")
		x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, source=True))
		return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
	"Mask out subsequent positions."
	attn_shape = (1, size, size)
	subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
	return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None, source=False):
	"Compute 'Scaled Dot Product Attention'"
	# print("Running Attention!", query.shape, key.shape, value.shape)
	d_k = query.size(-1)
	
	# Q . K^T / (root(d_k))
	scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)    
	
	# print("Printing from Scaled Dot Product Attention.")
	# print("######## Scores #########")
	# print(scores.shape,scores)
	# print("######## Mask ###########")
	# print(mask.shape,mask)
	
	if mask is not None:
		if source:
			scores = scores.masked_fill(mask.squeeze(-1) == 0, -1e9)
		else:
			scores = scores.masked_fill(mask == 0, -1e9)
	p_attn = F.softmax(scores, dim = -1)
	if dropout is not None:
		p_attn = dropout(p_attn)
	return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		"Take in model size and number of heads."
		super(MultiHeadedAttention, self).__init__()
		assert d_model % h == 0
		# We assume d_v always equals d_k
		self.d_k = d_model // h
		self.h = h
		self.linears = clones(nn.Linear(d_model, d_model), 4)
		self.attn = None
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, query, key, value, mask=None, source=False):
		"Implements Figure 2"
		if mask is not None:
			# Same mask applied to all h heads.
			mask = mask.unsqueeze(1)
		nbatches = query.size(0)
		
		# 1) Do all the linear projections in batch from d_model => h x d_k 
		query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
		
		# 2) Apply attention on all the projected vectors in batch. 
		x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout, source=source)
		
		# 3) "Concat" using a view and apply a final linear. 
		x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
		return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
	"Implements FFN equation."
	def __init__(self, d_model, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(d_model, d_ff)
		self.w_2 = nn.Linear(d_ff, d_model)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
	def __init__(self, d_model, vocab):
		super(Embeddings, self).__init__()
		self.lut = nn.Embedding(vocab, d_model)
		self.d_model = d_model

	def forward(self, x):
		print(x)
		return self.lut(x) * math.sqrt(self.d_model)

class EmbeddingLayer(nn.Module):
	def __init__(self, input_size, output_size):
		super(EmbeddingLayer, self).__init__()
		self.embed_layer = nn.Linear(input_size,output_size)
	
	def forward(self, x):
		return self.embed_layer(x)

class PositionalEncoding(nn.Module):
	"Implement the PE function."
	def __init__(self, d_model, dropout, max_len=5000):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)
		
		# Compute the positional encodings once in log space.
		pe = torch.zeros(max_len, d_model).cuda()
		position = torch.arange(0.0, max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)
		self.register_buffer('pe', pe)
			   
	def forward(self, x):
		# First dim is just batch.
		# Second dim is number of timesteps.
		# Third dim is d_model.
		x = x + Variable(self.pe[:, :x.size(0), :], requires_grad=False)
		return self.dropout(x)

class MultiDimensionalBatch():
	def __init__(self, source, target=None, pad=None):
		# Assume source and target are numpy arrays of size: Timesteps x Dimensions.
		if pad is None:
			self.pad = torch.zeros(source.shape[-1]).float().cuda()
		else:
			self.pad = torch.from_numpy(np.array(pad)).float().cuda()
		
		if type(source) is np.ndarray:
			self.source = torch.from_numpy(source).float().cuda()
		else:
			self.source = source
			
		# Case mask to float, since source and target themselves will be float.
		self.source_mask = 1.-(self.source==self.pad).all(dim=1).float()
		
		if target is not None:
			if type(target) is np.ndarray:
				self.target = torch.from_numpy(target).float().cuda()
			else:
				self.target = target.float().cuda()
			
			self.target_pad = torch.zeros(target.shape[-1]).float().cuda()

			self.target_mask = torch.ones((1,self.target.shape[0],self.target.shape[0])).cuda()
			# print(self.target.shape,self.target_mask.shape)
			self.target_mask_indices = 1.-(self.target==self.target_pad).all(dim=1).float()
			# print(self.target_mask_indices)
			self.target_mask[:,:,0] = 0.
			for i in range(self.target.shape[0]):            
				if self.target_mask_indices[i]==0.:
					self.target_mask[:,:,i] = 0.
				self.target_mask[:,:i,i] = 0.    

def sample_action(action_probabilities):
	# Categorical distribution sampling. 
	sample_action = torch.distributions.Categorical(probs=action_probabilities).sample().squeeze(0).cuda()
	return sample_action