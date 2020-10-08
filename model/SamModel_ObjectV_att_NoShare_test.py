import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import tensorflow as tf

import numpy as np
import math

rng = np.random
rng.seed(1234)

def hard_sigmoid(x):
	x = (0.2 * x) + 0.5
	x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),tf.cast(1., dtype=tf.float32))
	return x

class SoftModel():

	def __init__(self, video_feature, obj_feature, input_captions, dec_x_t, dec_h_tm1, v_encoder_output, o_encoder_output, 
		voc_size, d_w2v, output_dim, 
		reduction_dim=512, centers_num=16, filter_size=1, stride=[1,1,1,1], pad='SAME', 
		done_token=3, max_len = 20, beamsearch_batchsize = 1, beam_size=5,
		attention_dim = 100, dropout=0.5,inner_activation='hard_sigmoid',
		activation='tanh', return_sequences=True, bottleneck=256):

		self.reduction_dim=reduction_dim

		print('SamModel_ObjectV_att_NoShare_test.py: __init__: obj_feature:', obj_feature.get_shape().as_list())
		## original obj_feature shape [batch, obj_num, timestep, c, h, w]
		self.obj_feature = tf.transpose(obj_feature,perm=[1,0,2,4,5,3]) #  [obj_num, batch, timestep, h, w, c]
		
		print('SamModel_ObjectV_att_NoShare_test.py: __init__: video_feature:', video_feature.get_shape().as_list())
		## original video_feature shape [batch, timestep, c, h, w]
		self.video_feature = tf.transpose(video_feature,perm=[0,1,3,4,2]) #  [batch, timestep, h, w, c]
		
		self.input_captions = input_captions # shape [batch, max_len]


		self.dec_x_t = dec_x_t
		self.dec_h_tm1 = dec_h_tm1
		self.v_encoder_output = v_encoder_output
		self.o_encoder_output = o_encoder_output

		self.voc_size = voc_size
		self.d_w2v = d_w2v

		self.output_dim = output_dim
		self.filter_size = filter_size
		self.stride = stride
		self.pad = pad

		self.centers_num = centers_num

		self.beam_size = beam_size

		assert(beamsearch_batchsize==1)
		self.batch_size = beamsearch_batchsize
		self.done_token = done_token
		self.max_len = max_len

		self.dropout = dropout

		self.inner_activation = inner_activation
		self.activation = activation
		self.return_sequences = return_sequences
		self.attention_dim = attention_dim


		self.enc_in_shape_video = self.video_feature.get_shape().as_list() # [batch, timestep, h, w, c(2048)]
		self.enc_in_shape_obj = self.obj_feature.get_shape().as_list() # [obj_num, batch, timestep, h, w, c(2048)]
		self.decoder_input_shape = self.input_captions.get_shape().as_list() # [batch, caplen]
		print('SamModel_ObjectV_att_NoShare_test.py: __init__: self.enc_in_shape_video:', self.enc_in_shape_video,
		 	'self.enc_in_shape_obj:', self.enc_in_shape_obj,
			'self.decoder_input_shape:', self.decoder_input_shape)
		self.obj_num = self.enc_in_shape_obj[0]
		self.batch_size_v = self.enc_in_shape_obj[1]
		self.timesteps = self.enc_in_shape_obj[2]
		self.vfmap_h = self.enc_in_shape_obj[3]
		self.vfmap_w = self.enc_in_shape_obj[4]
		self.feature_dim = self.enc_in_shape_obj[5]


		self.bottleneck = bottleneck

	def init_parameters(self):
		print('init_parameters ...')
		
		############## encoder parameters
		## dimension reduction for input video feature: 2048 --> 512
		self.redu_W = tf.get_variable("redu_W", shape=[1, 1, self.feature_dim, self.reduction_dim], 
										initializer=tf.contrib.layers.xavier_initializer())
		self.redu_b = tf.get_variable("redu_b",initializer=tf.random_normal([self.reduction_dim],stddev=1./math.sqrt(self.reduction_dim)))
		
		###### for video feature
		## conv parameters for conv2d(W_e, x_t)
		self.W_e = tf.get_variable("W_e", shape=[3, 3, self.reduction_dim, 3*self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
		self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([3*self.centers_num],stddev=1./math.sqrt(3*self.centers_num)))
		## VLAD centers
		self.f_centers = tf.get_variable("f_centers",[1, 1, 1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.feature_dim)))
		## conv parameters for SC-GRU
		encoder_h2h_shape = (self.filter_size, self.filter_size, self.centers_num, self.centers_num) #[1,1,centers_num,centers_num]
		self.U_e_r = tf.get_variable("U_e_r", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_z = tf.get_variable("U_e_z", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_h = tf.get_variable("U_e_h", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())

		###### for object feature
		## conv parameters for conv2d(W_e, x_t)
		self.o_W_e = tf.get_variable("o_W_e", shape=[3, 3, self.reduction_dim, 3*self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
		self.o_b_e = tf.get_variable("o_b_e",initializer=tf.random_normal([3*self.centers_num],stddev=1./math.sqrt(3*self.centers_num)))
		## VLAD centers
		self.o_f_centers = tf.get_variable("o_f_centers",[1, 1, 1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.feature_dim)))
		## conv parameters for SC-GRU
		encoder_h2h_shape = (self.filter_size, self.filter_size, self.centers_num, self.centers_num) #[1,1,centers_num,centers_num]
		self.o_U_e_r = tf.get_variable("o_U_e_r", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.o_U_e_z = tf.get_variable("o_U_e_z", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.o_U_e_h = tf.get_variable("o_U_e_h", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer()) 
		
		########
		## linear parameters for mapping self.obj_feature to output as the initial hidden states of decoder
		if self.output_dim!=self.feature_dim:
			print("$$$init_parameters$$$       output_dim:",self.output_dim,' input feature dim:',self.feature_dim)#new
			#print('the dimension of input feature != hidden size')
			self.liner_W = tf.get_variable("liner_W",[self.feature_dim, self.output_dim],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.feature_dim)))
			self.liner_b = tf.get_variable("liner_b",initializer=tf.random_normal([self.output_dim],stddev=1./math.sqrt(self.output_dim)))

			self.o_liner_W = tf.get_variable("o_liner_W",[self.feature_dim, self.output_dim],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.feature_dim)))
			self.o_liner_b = tf.get_variable("o_liner_b",initializer=tf.random_normal([self.output_dim],stddev=1./math.sqrt(self.output_dim)))


		############## decoder parameters
		self.T_w2v, self.T_mask = self.init_embedding_matrix()

		decoder_i2h_shape = (self.d_w2v,3*self.output_dim)
		decoder_h2h_shape = (self.output_dim,self.output_dim)

		self.W_d = tf.get_variable("W_d",decoder_i2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.d_w2v)))
		self.b_d = tf.get_variable("b_d",initializer = tf.random_normal([3*self.output_dim], stddev=1./math.sqrt(3*self.output_dim)))
		
		self.U_d_r = tf.get_variable("U_d_r",decoder_h2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
		self.U_d_z = tf.get_variable("U_d_z",decoder_h2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
		self.U_d_h = tf.get_variable("U_d_h",decoder_h2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))

		###### attention parameters for video feature
		self.W_a = tf.get_variable("W_a",[self.reduction_dim*self.centers_num,self.attention_dim],
			initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.reduction_dim*self.centers_num)))
		#self.W_a and self.b_a for vlad
		#self.U_a for hidden states
		self.U_a = tf.get_variable("U_a",[self.output_dim,self.attention_dim],initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
		self.b_a = tf.get_variable("b_a",initializer = tf.random_normal([self.attention_dim],stddev=1. / math.sqrt(self.attention_dim))) 
		# self.W: mapping attention vector to a real value: attention_dim --> 1
		self.W = tf.get_variable("W",(self.attention_dim,1),initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.attention_dim)))

		###### attention parameters for object feature
		self.o_W_a = tf.get_variable("o_W_a",[self.reduction_dim*self.centers_num,self.attention_dim],
			initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.reduction_dim*self.centers_num)))
		self.o_U_a = tf.get_variable("o_U_a",[self.output_dim,self.attention_dim],initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
		self.o_b_a = tf.get_variable("o_b_a",initializer = tf.random_normal([self.attention_dim],stddev=1. / math.sqrt(self.attention_dim)))
		self.o_W = tf.get_variable("o_W",(self.attention_dim,1),initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.attention_dim)))

		### object attention parameters
		self.W_a_obj = tf.get_variable("W_a_obj",[self.reduction_dim*self.centers_num,self.attention_dim],
			initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.reduction_dim*self.centers_num)))
		self.U_a_obj = tf.get_variable("U_a_obj",[self.output_dim,self.attention_dim],initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
		self.b_a_obj = tf.get_variable("b_a_obj",initializer = tf.random_normal([self.attention_dim],stddev=1. / math.sqrt(self.attention_dim))) 
		
		# self.W_obj: mapping attention vector to a real value: attention_dim --> 1
		self.W_obj = tf.get_variable("W_obj",(self.attention_dim,1),initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.attention_dim)))

		######
		# self.A: parameters for mapping vlad in decoder: [A_z;A_r;A_h]
		self.A = tf.get_variable("A",(self.reduction_dim*self.centers_num,3*self.output_dim),
			initializer=tf.random_normal_initializer(stddev=1./ math.sqrt(self.reduction_dim*self.centers_num)))
		# self.o_A: parameters for mapping vlad in decoder: [A_z;A_r;A_h]
		self.o_A = tf.get_variable("o_A",(self.reduction_dim*self.centers_num,3*self.output_dim),
			initializer=tf.random_normal_initializer(stddev=1./ math.sqrt(self.reduction_dim*self.centers_num)))
		
		######
		# classification parameters
		self.W_c = tf.get_variable("W_c",[self.output_dim,self.voc_size],
			initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.output_dim)))
		self.b_c = tf.get_variable("b_c",initializer = tf.random_normal([self.voc_size],stddev=1./math.sqrt(self.voc_size)))

		
	def init_embedding_matrix(self):

		###init word embedding matrix
		
		#print('SamModel_ObjectV_att_NoShare_test.py: init_embedding_matrix starts ...')
		voc_size = self.voc_size # 12597
		d_w2v = self.d_w2v	# 512
		print('SamModel_ObjectV_att_NoShare_test.py: init_embedding_matrix: voc_size=', voc_size, 'd_w2v=', d_w2v)

		np_mask = np.vstack((np.zeros(d_w2v),np.ones((voc_size-1,d_w2v)))) # [voc_size, d_w2v]
		T_mask = tf.constant(np_mask, tf.float32, name='LUT_mask') # lookup table mask: the first row is all 0, the others are all 1
		print('SamModel_ObjectV_att_NoShare_test.py: init_embedding_matrix: T_mask:',T_mask.get_shape().as_list())

		LUT = np.zeros((voc_size, d_w2v), dtype='float32') # lookup table: shape [voc_size, d_w2v], the first row is all 0 for word ''
		for v in range(voc_size):
			LUT[v] = rng.randn(d_w2v)
			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

		# word 0 is blanked out, word 1 is 'UNK'
		LUT[0] = np.zeros((d_w2v))
		# setup LUT!
		T_w2v = tf.Variable(LUT.astype('float32'),trainable=True)
		print('SamModel_ObjectV_att_NoShare_test.py: init_embedding_matrix: T_w2v:',T_w2v.get_shape().as_list())
		#print('SamModel_ObjectV_att_NoShare_test.py: init_embedding_matrix ends ...')
		return T_w2v, T_mask 

	def encoder(self, input_feature): ### for video feature
		print('building encoder for video feature...')
		# input_feature (self.video_feature): shape [batch, timestep, h, w, c]
		timesteps = self.timesteps #number of frames of a video
		
		#### dimension reduction
		input_feature = tf.reshape(input_feature, [-1,self.vfmap_h,self.vfmap_w,self.feature_dim])
		t_ori_feature = input_feature # shape [batch*timestep, h, w, c]

		## feature dimension reduction
		input_feature = tf.add(tf.nn.conv2d(input_feature, self.redu_W, self.stride, self.pad, name='reduction_wx'),
			tf.reshape(self.redu_b,[1, 1, 1, self.reduction_dim]))
		
		input_feature = tf.reshape(input_feature,[-1,self.timesteps,self.vfmap_h,self.vfmap_w,self.reduction_dim]) # shape [batch, timestep, h, w, redu_dim]
		input_feature = tf.nn.relu(input_feature)
		self.enc_in_shape_video[-1] = input_feature.get_shape().as_list()[-1] # update self.enc_in_shape_video [batch, timestep, h, w, redu_dim]
																														# !!!Note: now the  self.enc_in_shape_video is not matched for self.video_feature with last dimension of 2048
		## to calculate conv2d(W, x_t) for SC-GRU
		assignment = tf.reshape(input_feature,[-1,self.vfmap_h,self.vfmap_w,self.reduction_dim])
		assignment = tf.add(tf.nn.conv2d(assignment, self.W_e, self.stride, self.pad, name='w_conv_x'),tf.reshape(self.b_e,[1, 1, 1, 3*self.centers_num]))
		assignment = tf.reshape(assignment,[-1,self.timesteps,self.vfmap_h,self.vfmap_w,3*self.centers_num]) # shape [batch, timestep, h, w, 3*centers_num]

		axis = [1,0]+list(range(2,5)) 
		assignment = tf.transpose(assignment, perm=axis) ## shape [timestep, batch, h, w, 3*centers_num]

		input_assignment = tf.TensorArray(
	            dtype=assignment.dtype,
	            size=timesteps,
	            tensor_array_name='input_assignment')
		if hasattr(input_assignment, 'unstack'):
			input_assignment = input_assignment.unstack(assignment)
		else:
			input_assignment = input_assignment.unpack(assignment)	
		## hidden states of SC-GRU
		hidden_states = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='hidden_states')

		def get_init_state(x, output_dims): # x = input_feature; output_dims = centers_num
			initial_state = tf.zeros_like(x)
			initial_state = tf.reduce_sum(initial_state,axis=[1,4]) # shape [batch, h, w]
			initial_state = tf.expand_dims(initial_state,dim=-1) # shape [batch, h, w, 1]
			initial_state = tf.tile(initial_state,[1,1,1,output_dims]) # shape [batch, h, w, centers_num]
			return initial_state
		def step(time, hidden_states, h_tm1): # hidden_states = hidden_states; h_tm1 = initial_state
			assign_t = input_assignment.read(time) # shape [batch, h, w, 3*centers_num]

			assign_t_r, assign_t_z, assign_t_h = tf.split(assign_t,3,axis=3) # shape [batch, h, w, centers_num]
			
			r = hard_sigmoid(assign_t_r+ tf.nn.conv2d(h_tm1, self.U_e_r, self.stride, self.pad, name='r'))
			z = hard_sigmoid(assign_t_z+ tf.nn.conv2d(h_tm1, self.U_e_z, self.stride, self.pad, name='z'))

			hh = tf.tanh(assign_t_h+ tf.nn.conv2d(r*h_tm1, self.U_e_h, self.stride, self.pad, name='uh_hh'))

			h = (1-z)*hh + z*h_tm1
			
			hidden_states = hidden_states.write(time, h)

			return (time+1,hidden_states, h)

		time = tf.constant(0, dtype='int32', name='time')
		initial_state = get_init_state(input_feature,self.centers_num) # input_feature: shape [batch, timestep, h, w, redu_dim]
																																	 # intial_state: tf.zeros, shape [batch, h, w, centers_num]

		feature_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=step,
	            loop_vars=(time, hidden_states, initial_state ),
	            parallel_iterations=32,
	            swap_memory=True)

		hidden_states = feature_out[-2]
		if hasattr(hidden_states, 'stack'):
			assignment = hidden_states.stack()
		else:
			assignment = hidden_states.pack() #shape [timestep, batch, h, w, centers_num]
		print('SamModel_ObjectV_att_NoShare_test.py: encoder: assignment:', assignment)
		axis = [1,0]+list(range(2,5)) 
		assignment = tf.transpose(assignment, perm=axis) #shape [batch, timestep, h, w, centers_num]

		assignment = tf.reshape(assignment,[-1,self.vfmap_h*self.vfmap_w,self.centers_num]) #shape [batch*timestep, h*w, centers_num]
		## edited by zjc 2018/08/25: to avoid FB seperation
		'''
		# backgroung,front
		f_assignment = threshold*assignment
		b_assignment = (1-threshold)*assignment
		'''
		def apart(assignment,input_feature,centers):
			a_sum = tf.reduce_sum(assignment,-2,keep_dims=True) # shape [batch*timestep, 1, centers_num]
			a = tf.multiply(a_sum,centers) # shape [batch*timestep, redu_dim, centers_num]
			assignment = tf.transpose(assignment,perm=[0,2,1]) # shape [batch*timestep, centers_num, h*w]

			input_feature = tf.reshape(input_feature,[-1,self.vfmap_h*self.vfmap_w,self.reduction_dim])
			# input_feature: shape [batch*timestep, h*w, redu_dim]
			vlad = tf.matmul(assignment,input_feature) # shape [batch*timestep, centers_num, redu_dim]
			vlad = tf.transpose(vlad, perm=[0,2,1]) # shape [batch*timestep, redu_dim, centers_num]
			tf.summary.histogram('vlad',vlad)
			
			# for differnce
			vlad = tf.subtract(vlad,a) # shape [batch*timestep, redu_dim, centers_num]
			vlad = tf.reshape(vlad,[-1,self.reduction_dim,self.centers_num]) # shape [batch*timestep, redu_dim, centers_num]
			vlad = tf.nn.l2_normalize(vlad,1)

			vlad = tf.reshape(vlad,[-1,self.timesteps,self.reduction_dim*self.centers_num]) #shape [batch, timestep, redu_dim*centers_num]
			vlad = tf.nn.l2_normalize(vlad,2)
			
			return vlad #shape [batch, timestep, redu_dim*centers_num]
		## zjc
		#f_vlad = apart(f_assignment,input_feature,self.f_centers)
		#b_vlad = apart(b_assignment,input_feature,self.f_centers)

		# self.f_centers: shape [1,1,1,redu_dim, centers_num]
		# input_feature: shape [batch, timestep, h, w, redu_dim]
		# assignment: shape [batch*timestep, h*w, centers_num]
		vlad = apart(assignment,input_feature,self.f_centers) #shape [batch, timestep, redu_dim*centers_num]
		'''
		last_output = tf.reduce_mean(self.obj_feature,axis=[0,2,3,4]) # shape [batch, c(2048)]
		if self.output_dim!=self.obj_feature.get_shape().as_list()[-1]:
			print '$$$$$apart$$$$   output_dim:', self.output_dim,' self.obj_feature:', self.obj_feature.get_shape().as_list()[-1] #new

                        print('the dimension of self.obj_feature != hidden size')
			last_output = tf.nn.xw_plus_b(last_output,self.liner_W, self.liner_b) # shape [batch, output_dim]
		
		#return last_output, f_vlad, b_vlad
		return last_output, vlad
		'''
		return vlad

	def encoder_allobjs(self, input_feature): ### for object feature
		print('building encoder for all objects... ...')
		# input_feature (self.obj_feature): shape [obj_num, batch, timestep, h, w, c]
		timesteps = self.timesteps #number of frames of a video
		obj_num = input_feature.get_shape()[0]
		
		###### dimension reduction
		input_feature = tf.reshape(input_feature,[-1,self.vfmap_h,self.vfmap_w,self.feature_dim])
		t_ori_feature = input_feature # shape [obj_num*batch*timestep, h, w, c]

		## feature dimension reduction
		input_feature = tf.add(tf.nn.conv2d(input_feature, self.redu_W, self.stride, self.pad, name='o_reduction_wx'),
			tf.reshape(self.redu_b,[1, 1, 1, self.reduction_dim]))

		input_feature = tf.reshape(input_feature,[-1,self.timesteps,self.vfmap_h,self.vfmap_w,self.reduction_dim]) 
		input_feature = tf.nn.relu(input_feature) # shape [obj_num*batch, timestep, h, w, redu_dim]
		self.enc_in_shape_obj[-1] = input_feature.get_shape().as_list()[-1] # update self.enc_in_shape_obj [obj_num, batch, timestep, h, w, redu_dim]
																														# !!!Note: now the  self.enc_in_shape_obj is not matched for self.obj_feature with last dimension of 2048
		## to calculate conv2d(W, x_t) for SC-GRU
		assignment = tf.reshape(input_feature,[-1,self.vfmap_h,self.vfmap_w,self.reduction_dim]) # shape [obj_num*batch*timestep, h, w, redu_dim]
		assignment = tf.add(tf.nn.conv2d(assignment, self.o_W_e, self.stride, self.pad, name='o_w_conv_x'),
			tf.reshape(self.o_b_e,[1, 1, 1, 3*self.centers_num]))
		assignment = tf.reshape(assignment,[-1,self.timesteps,self.vfmap_h,self.vfmap_w,3*self.centers_num])
		# assignment: shape [obj_num*batch, timestep, h, w, 3*centers_num]

		axis = [1,0]+list(range(2,5)) 
		assignment = tf.transpose(assignment, perm=axis) ## shape [timestep, obj_num*batch, h, w, 3*centers_num]

		input_assignment = tf.TensorArray(
	            dtype=assignment.dtype,
	            size=timesteps,
	            tensor_array_name='o_input_assignment')
		if hasattr(input_assignment, 'unstack'):
			input_assignment = input_assignment.unstack(assignment)
		else:
			input_assignment = input_assignment.unpack(assignment)	
		## hidden states of SC-GRU
		hidden_states = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='o_hidden_states')

		def get_init_state(x, output_dims): # x = input_feature; output_dims = centers_num
			initial_state = tf.zeros_like(x)
			initial_state = tf.reduce_sum(initial_state,axis=[1,4]) # shape [obj_num*batch, h, w]
			initial_state = tf.expand_dims(initial_state,dim=-1) # shape [obj_num*batch, h, w, 1]
			initial_state = tf.tile(initial_state,[1,1,1,output_dims]) # shape [obj_num*batch, h, w, centers_num]
			return initial_state
		def step(time, hidden_states, h_tm1): # hidden_states = hidden_states; h_tm1 = initial_state
			assign_t = input_assignment.read(time) # shape [obj_num*batch, h, w, 3*centers_num]

			assign_t_r, assign_t_z, assign_t_h = tf.split(assign_t,3,axis=3) # shape [obj_num*batch, h, w, centers_num]
			
			r = hard_sigmoid(assign_t_r+ tf.nn.conv2d(h_tm1, self.o_U_e_r, self.stride, self.pad, name='o_r'))
			z = hard_sigmoid(assign_t_z+ tf.nn.conv2d(h_tm1, self.o_U_e_z, self.stride, self.pad, name='o_z'))

			hh = tf.tanh(assign_t_h+ tf.nn.conv2d(r*h_tm1, self.o_U_e_h, self.stride, self.pad, name='o_uh_hh'))

			h = (1-z)*hh + z*h_tm1
			
			hidden_states = hidden_states.write(time, h)

			return (time+1,hidden_states, h)

		time = tf.constant(0, dtype='int32', name='o_time')
		initial_state = get_init_state(input_feature,self.centers_num) # input_feature: shape [obj_num*batch, timestep, h, w, redu_dim]
																																		# intial_state: tf.zeros, shape [obj_num*batch, h, w, centers_num]

		feature_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=step,
	            loop_vars=(time, hidden_states, initial_state ),
	            parallel_iterations=32,
	            swap_memory=True)

		hidden_states = feature_out[-2]
		if hasattr(hidden_states, 'stack'):
			assignment = hidden_states.stack()
		else:
			assignment = hidden_states.pack() #shape [timestep, obj_num*batch, h, w, centers_num]
		print('SamModel_ObjectV_att_NoShare_test.py: encoder_allobjs: assignment:', assignment)
		axis = [1,0]+list(range(2,5)) 
		assignment = tf.transpose(assignment, perm=axis) #shape [obj_num*batch, timestep, h, w, centers_num]

		assignment = tf.reshape(assignment,[-1,self.vfmap_h*self.vfmap_w,self.centers_num]) #shape [obj_num*batch*timestep, h*w, centers_num]
		
		def apart(assignment,input_feature,centers):
			a_sum = tf.reduce_sum(assignment,-2,keep_dims=True) # shape [obj_num*batch*timestep, 1, centers_num]
			a = tf.multiply(a_sum,centers) # shape [obj_num*batch*timestep, redu_dim, centers_num]
			assignment = tf.transpose(assignment,perm=[0,2,1]) # shape [obj_num*batch*timestep, centers_num, h*w]

			input_feature = tf.reshape(input_feature,[-1,self.vfmap_h*self.vfmap_w,self.reduction_dim])
			# input_feature: shape [obj_num*batch*timestep, h*w, redu_dim]
			vlad = tf.matmul(assignment,input_feature) # shape [obj_num*batch*timestep, centers_num, redu_dim]
			vlad = tf.transpose(vlad, perm=[0,2,1]) # shape [obj_num*batch*timestep, redu_dim, centers_num]
			tf.summary.histogram('vlad',vlad)
			
			# for differnce
			vlad = tf.subtract(vlad,a) # shape [obj_num*batch*timestep, redu_dim, centers_num]
			vlad = tf.reshape(vlad,[-1,self.reduction_dim,self.centers_num]) # shape [obj_num*batch*timestep, redu_dim, centers_num]
			vlad = tf.nn.l2_normalize(vlad,1)

			vlad = tf.reshape(vlad,[-1,self.timesteps,self.reduction_dim*self.centers_num]) #shape [obj_num*batch, timestep, redu_dim*centers_num]
			vlad = tf.nn.l2_normalize(vlad,2)
			
			return vlad #shape [obj_num*batch, timestep, redu_dim*centers_num]
		
		# self.f_centers: shape [1,1,1,redu_dim, centers_num]
		# input_feature: shape [obj_num*batch, timestep, h, w, redu_dim]
		# assignment: shape [obj_num*batch*timestep, h*w, centers_num]
		vlad = apart(assignment,input_feature,self.o_f_centers) #shape [obj_num*batch, timestep, redu_dim*centers_num]
		'''
		last_output = tf.reduce_mean(self.obj_feature,axis=[0,2,3,4]) # shape [batch, c(2048)]
		if self.output_dim!=self.obj_feature.get_shape().as_list()[-1]:
			print '$$$$$apart$$$$   output_dim:', self.output_dim,' self.obj_feature:', self.obj_feature.get_shape().as_list()[-1] #new

                        print('the dimension of self.obj_feature != hidden size')
			last_output = tf.nn.xw_plus_b(last_output,self.liner_W, self.liner_b) # shape [batch, output_dim]
		
		#return last_output, f_vlad, b_vlad
		return last_output, vlad
		'''
		vlad = tf.reshape(vlad, [obj_num, -1, self.timesteps, self.reduction_dim*self.centers_num])
		return vlad


	def decoder_objV_att(self, initial_state, input_feature, o_input_feature):
		# initial_state: shape [batch, output_dim]: the output_dim is the dimension of decoder hidden states
		# input_feature = v_vlad: [batch, timestep, redu_dim*centers_num]
		# o_input_feature = obj_vlads: [obj_num, batch, timestep, redu_dim*centers_num]
		# captions: [batch, caplen] ,int32
		# d_w2v: dimension of w2v

		print('up building decoder with object attention ... ...')
		captions = self.input_captions # shape [batch, caplen]
		mask =  tf.not_equal(captions,0) # shape [batch, caplen], bool: to indicate non-blank words
		loss_mask = tf.cast(mask,tf.float32) # shape [batch, caplen], float32

		## convert words to embeddings
		embedded_captions = tf.gather(self.T_w2v,captions)*tf.gather(self.T_mask,captions) # shape [batch, caplen, d_w2v]
		timesteps = self.decoder_input_shape[1] # =caplen
		axis = [1,0]+list(range(2,3)) # [1,0,2]
		embedded_captions = tf.transpose(embedded_captions, perm=axis) # shape [caplen, batch, d_w2v] 
		input_embedded_words = tf.TensorArray(
	            dtype=embedded_captions.dtype,
	            size=timesteps,
	            tensor_array_name='input_embedded_words')
		if hasattr(input_embedded_words, 'unstack'):
			input_embedded_words = input_embedded_words.unstack(embedded_captions)
		else:
			input_embedded_words = input_embedded_words.unpack(embedded_captions)	

		# preprocess mask
		mask = tf.expand_dims(mask,dim=-1) # shape [batch, caplen, 1]
		
		mask = tf.transpose(mask,perm=axis) # shape [caplen, batch, 1]

		input_mask = tf.TensorArray(
			dtype=mask.dtype,
			size=timesteps,
			tensor_array_name='input_mask'
			)
		if hasattr(input_mask, 'unstack'):
			input_mask = input_mask.unstack(mask)
		else:
			input_mask = input_mask.unpack(mask)


		train_hidden_state = tf.TensorArray( # caplen * [batch, output_dim]
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='train_hidden_state')

		def step(x_t,h_tm1):
			## x_t: shape [batch, d_w2v], h_tm1: shape [batch, output_dim]
			## here the input_feature indicates v_vlad, shape [batch, timestep(encoder), redu_dim*centers_num]
			## here the o_input_feature indicates obj_vlad, shape [obj_num*batch, timestep(encoder), redu_dim*centers_num]
			ori_feature = tf.reshape(input_feature,(-1,self.reduction_dim*self.centers_num)) 
			# ori_feature: shape [batch*timestep(encoder), redu_dim*centers_num]
			o_ori_feature = tf.reshape(o_input_feature,(-1,self.reduction_dim*self.centers_num)) 
			# o_ori_feature: shape [obj_num*batch*timestep(encoder), redu_dim*centers_num]
			
			########
			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.timesteps,self.attention_dim))
			# attend_wx: shape [batch, timestep(encoder), attention_dim]
			attend_uh_tm1 = tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1) # shape [batch, 1, attention_dim]
			attend_e = tf.nn.tanh(tf.add(attend_wx,attend_uh_tm1)) # shape [batch, timestep(encoder), attention_dim]
			
			o_attend_wx = tf.reshape(tf.nn.xw_plus_b(o_ori_feature, self.o_W_a, self.o_b_a),(-1,self.timesteps,self.attention_dim))
			# attend_wx: shape [obj_num*batch, timestep(encoder), attention_dim]
			o_attend_uh_tm1 = tf.expand_dims(tf.matmul(h_tm1, self.o_U_a),dim=1)
			o_attend_uh_tm1 = tf.tile(o_attend_uh_tm1, [self.obj_num, 1, 1]) # shape [obj_num*batch, 1, attention_dim]
			o_attend_e = tf.nn.tanh(tf.add(o_attend_wx, o_attend_uh_tm1)) # shape [obj_num*batch, timestep(encoder), attention_dim]
			
			########
			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W) # shape [batch*timestep(encoder)]
			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.timesteps,1)),dim=1) # shape [batch, timestep(encoder), 1]
			attend_e = tf.reshape(attend_e,(-1,self.timesteps,1)) # shape [batch, timestep(encoder), 1]

			o_attend_e = tf.matmul(tf.reshape(o_attend_e,(-1,self.attention_dim)),self.o_W) # shape [obj_num*batch*timestep(encoder)]
			o_attend_e = tf.nn.softmax(tf.reshape(o_attend_e,(-1,self.timesteps,1)),dim=1) # shape [obj_num*batch, timestep(encoder), 1]
			o_attend_e = tf.reshape(o_attend_e,(-1,self.timesteps,1)) # shape [(1+obj_num)*batch, timestep(encoder), 1]

			########
			attend_fea = tf.multiply(tf.reshape(input_feature,(-1,self.timesteps,self.reduction_dim*self.centers_num)) , attend_e) 
			#attend_fea: shape [batch, timestep(encoder), redu_dim*centers_num]
			attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1) # shape [batch, redu_dim*centers_num]

			o_attend_fea = tf.multiply(tf.reshape(o_input_feature,(-1,self.timesteps,self.reduction_dim*self.centers_num)) , o_attend_e) 
			#o_attend_fea: shape [obj_num*batch, timestep(encoder), redu_dim*centers_num]
			o_attend_fea = tf.reduce_sum(o_attend_fea,reduction_indices=1) # shape [obj_num*batch, redu_dim*centers_num]

			########
			### for object attention
			o_attend_fea = tf.reshape(o_attend_fea, [self.obj_num, -1, self.reduction_dim*self.centers_num]) # shape [obj_num, batch, redu_dim*centers_num]
			o_attend_fea = tf.transpose(o_attend_fea, perm=[1, 0, 2]) # shape [batch, obj_num, redu_dim*centers_num]
			ori_o_attend_fea = tf.reshape(o_attend_fea, [-1, self.reduction_dim*self.centers_num]) # shape [batch*obj_num, redu_dim*centers_num]

			attend_wx_obj = tf.reshape(tf.nn.xw_plus_b(ori_o_attend_fea, self.W_a_obj, self.b_a_obj),(-1,self.obj_num,self.attention_dim))
			# attend_wx_obj: shape [batch, obj_num, attention_dim]
			attend_uh_tm1_obj = tf.expand_dims(tf.matmul(h_tm1, self.U_a_obj),dim=1) # shape [batch, 1, attention_dim]
			attend_e_obj = tf.nn.tanh(tf.add(attend_wx_obj, attend_uh_tm1_obj)) # # shape [batch, obj_num, attention_dim]

			attend_e_obj = tf.matmul(tf.reshape(attend_e_obj,(-1,self.attention_dim)),self.W_obj) # shape [batch*obj_num]
			attend_e_obj = tf.nn.softmax(tf.reshape(attend_e_obj,(-1,self.obj_num,1)),dim=1) # shape [batch, obj_num, 1]
			attend_e_obj = tf.reshape(attend_e_obj,(-1,self.obj_num,1)) # shape [batch, obj_num, 1]

			attend_fea_obj = tf.multiply(tf.reshape(o_attend_fea, (-1,self.obj_num,self.reduction_dim*self.centers_num)), attend_e_obj) 
			#attend_fea_obj: shape [batch, obj_num, redu_dim*centers_num]
			o_attend_fea = tf.reduce_sum(attend_fea_obj, reduction_indices=1) # shape [batch, redu_dim*centers_num]

			########
			attend_fea = tf.add(tf.matmul(attend_fea,self.A),tf.matmul(o_attend_fea, self.o_A)) # shape [batch, 3*output_dim]

			attend_fea_r = attend_fea[:,0:self.output_dim]
			attend_fea_z = attend_fea[:,self.output_dim:2*self.output_dim]
			attend_fea_h = attend_fea[:,2*self.output_dim::]

			preprocess_x = tf.nn.xw_plus_b(x_t, self.W_d, self.b_d)

			preprocess_x_r = preprocess_x[:,0:self.output_dim]
			preprocess_x_z = preprocess_x[:,self.output_dim:2*self.output_dim]
			preprocess_x_h = preprocess_x[:,2*self.output_dim::]

			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_d_r) + attend_fea_r)
			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_d_z) + attend_fea_z)
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_d_h) + attend_fea_h)
			
			h = (1-z)*hh + z*h_tm1

			return h #shape [batch, output_dim]

		def train_step(time, train_hidden_state, h_tm1):
			x_t = input_embedded_words.read(time) # shape [batch, d_w2v]
			mask_t = input_mask.read(time) # shape [batch, 1]

			h = step(x_t,h_tm1) # shape [batch, output_dim]

			tiled_mask_t = tf.tile(mask_t, tf.stack([1, h.get_shape().as_list()[1]])) # shape [batch, output_dim]

			h = tf.where(tiled_mask_t, h, h_tm1) # shape [batch, output_dim]
			
			train_hidden_state = train_hidden_state.write(time, h)

			return (time+1,train_hidden_state,h)

		time = tf.constant(0, dtype='int32', name='time')

		train_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=train_step,
	            loop_vars=(time, train_hidden_state, initial_state), # initial_state indicates the h_tm1, shape [batch, output_dim]
	            parallel_iterations=32,
	            swap_memory=True)

		train_hidden_state = train_out[1]
		train_last_output = train_out[-1] # hidden states at the last timestep
		
		if hasattr(train_hidden_state, 'stack'):
			train_outputs = train_hidden_state.stack()
		else:
			train_outputs = train_hidden_state.pack() # shape [caplen, batch, output_dim]

		axis = [1,0] + list(range(2,3)) # [1, 0, 2]
		train_outputs = tf.transpose(train_outputs,perm=axis) # shape [batch, caplen, output_dim]

		train_outputs = tf.reshape(train_outputs,(-1,self.output_dim)) # shape [batch*caplen, output_dim]
		train_outputs = tf.nn.dropout(train_outputs, self.dropout)
		predict_score = tf.matmul(train_outputs,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size)) # shape [batch*caplen, voc_size]
		predict_score = tf.reshape(predict_score,(-1,timesteps,self.voc_size)) # shape [batch, caplen, voc_size]

		## for test_step: predict words using gready search
		test_input_embedded_words = tf.TensorArray(
	            dtype=embedded_captions.dtype,
	            size=timesteps+1,
	            tensor_array_name='test_input_embedded_words')

		predict_words = tf.TensorArray(
	            dtype=tf.int64,
	            size=timesteps,
	            tensor_array_name='predict_words')

		test_hidden_state = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='test_hidden_state')
		test_input_embedded_words = test_input_embedded_words.write(0,embedded_captions[0])

		def test_step(time, test_hidden_state, test_input_embedded_words, predict_words, h_tm1):
			x_t = test_input_embedded_words.read(time) # shape [batch, d_w2v]

			h = step(x_t,h_tm1) # shape [batch, output_dim]

			test_hidden_state = test_hidden_state.write(time, h)

			drop_h = h*self.dropout
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size)) # shape [batch, voc_size]

			predict_score_t = tf.nn.softmax(predict_score_t,dim=-1) # shape [batch, voc_size]
			predict_word_t = tf.argmax(predict_score_t,-1) # shape [batch,1]

			predict_words = predict_words.write(time, predict_word_t) # output

			predict_word_t = tf.gather(self.T_w2v,predict_word_t)*tf.gather(self.T_mask,predict_word_t) # shape [batch, d_w2v]

			test_input_embedded_words = test_input_embedded_words.write(time+1,predict_word_t)

			return (time+1,test_hidden_state, test_input_embedded_words, predict_words, h)

		time = tf.constant(0, dtype='int32', name='time')

		test_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=test_step,
	            loop_vars=(time, test_hidden_state, test_input_embedded_words, predict_words, initial_state),
	            # the initial_state is the last_output from encoder: mapping of the mean of the self.obj_feature 
	            parallel_iterations=32,
	            swap_memory=True)

		predict_words = test_out[-2]
		
		if hasattr(predict_words, 'stack'):
			predict_words = predict_words.stack()
		else:
			predict_words = predict_words.pack() # shape [caplen, batch, 1]

		axis = [1,0] + list(range(2,3)) # [1, 0, 2]

		predict_words = tf.transpose(predict_words,perm=[1,0]) # shape [batch, caplen, 1]
		predict_words = tf.reshape(predict_words,(-1,timesteps)) # shape [batch, caplen]
		## predict_score: shape [batch, caplen, voc_size], scores for train_hidden_states
		## predict_words: shape [batch, caplen], for test hidden_states
		## loss_mask: shape [batch, caplen], float32, the float version of mask (bool)
		return predict_score, predict_words, loss_mask

	def beamSearchDecoder_objV_att(self, initial_state, input_feature, o_input_feature):

		def step(x_t,h_tm1):
			# x_t: shape [beam_size, d_w2v]
			# h_tm1: shape [beam_size, output_dim]
			# here the input_feature indicates v_vlads, shape [1(beam_batch), timestep(encoder), redu_dim*centers_num]
			# here the o_input_feature indicates obj_vlads, shape [obj_num, 1(beam_batch), timestep(encoder), redu_dim*centers_num]
			print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: step: input_feature:', input_feature, 'o_input_feature:', o_input_feature)
			this_input_feature = tf.tile(input_feature, [self.beam_size,1,1])
			o_this_input_feature = tf.tile(o_input_feature, [1,self.beam_size,1,1])
			
			ori_feature = tf.reshape(this_input_feature,(-1,self.reduction_dim*self.centers_num))
			#ori_feature: shape [beam_size*timestep(encoder), redu_dim*centers_num]
			o_ori_feature = tf.reshape(o_this_input_feature,(-1,self.reduction_dim*self.centers_num))
			#o_ori_feature: shape [obj_num*beam_size*timestep(encoder), redu_dim*centers_num]
			
			#########
			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.timesteps,self.attention_dim))
			# attend_wx: shape [beam_size, timestep(encoder), attention_dim]
			attend_uh_tm1 = tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1) # shape [beam_size, 1, attention]
			attend_e = tf.nn.tanh(tf.add(attend_wx,attend_uh_tm1)) # shape [beam_size, timestep, attention]

			o_attend_wx = tf.reshape(tf.nn.xw_plus_b(o_ori_feature, self.o_W_a, self.o_b_a),(-1,self.timesteps,self.attention_dim))
			# o_attend_wx: shape [obj_num*beam_size, timestep(encoder), attention_dim]
			o_attend_uh_tm1 = tf.expand_dims(tf.matmul(h_tm1, self.o_U_a),dim=1) # shape [beam_size, 1, attention]
			o_attend_uh_tm1 = tf.tile(o_attend_uh_tm1, [self.obj_num, 1, 1]) # shape [obj_num*beam_size, 1, attention]
			o_attend_e = tf.nn.tanh(tf.add(o_attend_wx,o_attend_uh_tm1))
			
			#########
			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W)
			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.timesteps,1)),dim=1)
			attend_e = tf.reshape(attend_e,(-1,self.timesteps,1))

			o_attend_e = tf.matmul(tf.reshape(o_attend_e,(-1,self.attention_dim)),self.o_W)
			o_attend_e = tf.nn.softmax(tf.reshape(o_attend_e,(-1,self.timesteps,1)),dim=1)
			o_attend_e = tf.reshape(o_attend_e,(-1,self.timesteps,1))
			
			#########
			attend_fea = tf.multiply(tf.reshape(this_input_feature,(-1,self.timesteps,self.reduction_dim*self.centers_num)) , attend_e) 
			# attend_fea: shape [beam_size, timestep(encoder), redu_dim*centers_num]
			attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1) # shape [beam_size, redu_dim*centers_num]

			o_attend_fea = tf.multiply(tf.reshape(o_this_input_feature,(-1,self.timesteps,self.reduction_dim*self.centers_num)) , o_attend_e) 
			# attend_fea: shape [obj_num*beam_size, timestep(encoder), redu_dim*centers_num]
			o_attend_fea = tf.reduce_sum(o_attend_fea,reduction_indices=1) # shape [obj_num*beam_size, redu_dim*centers_num]
			
			#########
			### for object attention
			o_attend_fea = tf.reshape(o_attend_fea, [self.obj_num, -1, self.reduction_dim*self.centers_num]) # shape [obj_num, beam_size, redu_dim*centers_num]
			o_attend_fea = tf.transpose(o_attend_fea, perm=[1, 0, 2]) # shape [beam_size, obj_num, redu_dim*centers_num]
			ori_o_attend_fea = tf.reshape(o_attend_fea, [-1, self.reduction_dim*self.centers_num]) # shape [beam_size*obj_num, redu_dim*centers_num]

			attend_wx_obj = tf.reshape(tf.nn.xw_plus_b(ori_o_attend_fea, self.W_a_obj, self.b_a_obj),(-1,self.obj_num,self.attention_dim))
			# attend_wx_obj: shape [beam_size, obj_num, attention_dim]
			attend_uh_tm1_obj = tf.expand_dims(tf.matmul(h_tm1, self.U_a_obj),dim=1) # shape [beam_size, 1, attention_dim]
			attend_e_obj = tf.nn.tanh(tf.add(attend_wx_obj, attend_uh_tm1_obj)) # # shape [beam_size, obj_num, attention_dim]

			attend_e_obj = tf.matmul(tf.reshape(attend_e_obj,(-1,self.attention_dim)),self.W_obj) # shape [beam_size*obj_num]
			attend_e_obj = tf.nn.softmax(tf.reshape(attend_e_obj,(-1,self.obj_num,1)),dim=1) # shape [beam_size, obj_num, 1]
			attend_e_obj = tf.reshape(attend_e_obj,(-1,self.obj_num,1)) # shape [beam_size, obj_num, 1]

			attend_fea_obj = tf.multiply(tf.reshape(o_attend_fea, (-1,self.obj_num,self.reduction_dim*self.centers_num)), attend_e_obj) 
			#attend_fea_obj: shape [beam_size, obj_num, redu_dim*centers_num]
			o_attend_fea = tf.reduce_sum(attend_fea_obj, reduction_indices=1) # shape [beam_size, redu_dim*centers_num]
			
			#########
			attend_fea = tf.add(tf.matmul(attend_fea,self.A),tf.matmul(o_attend_fea, self.o_A)) # shape [beam_size, 3*output_dim]


			attend_fea_r = attend_fea[:,0:self.output_dim]
			attend_fea_z = attend_fea[:,self.output_dim:2*self.output_dim]
			attend_fea_h = attend_fea[:,2*self.output_dim::]

			preprocess_x = tf.nn.xw_plus_b(x_t, self.W_d, self.b_d)

			preprocess_x_r = preprocess_x[:,0:self.output_dim]
			preprocess_x_z = preprocess_x[:,self.output_dim:2*self.output_dim]
			preprocess_x_h = preprocess_x[:,2*self.output_dim::]

			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_d_r) + attend_fea_r)
			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_d_z) + attend_fea_z)
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_d_h) + attend_fea_h)
			
			h = (1-z)*hh + z*h_tm1

			return h

		def take_step_zero(x_0, h_0):
			x_0 = tf.gather(self.T_w2v,x_0)*tf.gather(self.T_mask,x_0)
			x_0 = tf.reshape(x_0,[self.batch_size*self.beam_size,self.d_w2v]) # shape [beam_size, d_w2v]
			
			h = step(x_0,h_0) # shape [beam_size, output_dim]
			
			drop_h = h # no dropout
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))
			logprobs = tf.nn.log_softmax(predict_score_t) # shape [beam_size, voc_size]

			logprobs_batched = tf.reshape(logprobs, [-1, self.beam_size, self.voc_size]) # shape [1(beam_batch), beam_size, voc_size]
			
			past_logprobs, indices = tf.nn.top_k(
			        logprobs_batched[:,0,:],self.beam_size)
			#past_logprobs: shape [1, beam_size]; indices: shape [1, beam_size]
			print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: take_step_zero: past_logprobs:', past_logprobs.get_shape().as_list(),
				'indices:', indices.get_shape().as_list())
			symbols = indices % self.voc_size # shape [1, beam_size]
			parent_refs = indices//self.voc_size # shape [1, beam_size]
			h = tf.gather(h,  tf.reshape(parent_refs,[-1])) # shape [1*beam_size, output_dim]
			
			past_symbols = tf.concat([tf.expand_dims(symbols, 2), tf.zeros((self.batch_size, self.beam_size, self.max_len-1), dtype=tf.int32)],-1)
			# past_symbols: shape [1, beam_size, caplen]
			return symbols, h, past_symbols, past_logprobs


		def test_step(time, x_t, h_tm1, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams):
			#s_t: symbols, shape [1, beam_size]
			#past_symbols: shape [1, beam_size, caplen]
			#past_logprobs: shape [1, beam_size]
			
			x_t = tf.gather(self.T_w2v,x_t)*tf.gather(self.T_mask,x_t)
			x_t = tf.reshape(x_t,[self.batch_size*self.beam_size,self.d_w2v])
			h = step(x_t,h_tm1)

			drop_h = h # no dropout
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

			logprobs = tf.nn.log_softmax(predict_score_t)
			logprobs = tf.reshape(logprobs, [1, self.beam_size, self.voc_size]) # shape [1(beam_batch), beam_size, voc_size]
			
			logprobs = logprobs+tf.expand_dims(past_logprobs, 2) # shape [1(beam_batch), beam_size, voc_size]
			past_logprobs, topk_indices = tf.nn.top_k(
			    tf.reshape(logprobs, [1, self.beam_size * self.voc_size]),
			    self.beam_size, 
			    sorted=False
			)       
			
			symbols = topk_indices % self.voc_size
			symbols = tf.reshape(symbols, [1,self.beam_size]) # shape [1, beam_size]
			parent_refs = topk_indices // self.voc_size

			h = tf.gather(h,  tf.reshape(parent_refs,[-1])) # shape [1, beam_size, output_dim]
			past_symbols_batch_major = tf.reshape(past_symbols[:,:,0:time], [-1, time])

			beam_past_symbols = tf.gather(past_symbols_batch_major,  parent_refs) # shape [1, beam_size, t-1]
			print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: test_step: beam_past_symbols:', beam_past_symbols.get_shape().as_list())
			past_symbols = tf.concat([beam_past_symbols, tf.expand_dims(symbols, 2), tf.zeros((1, self.beam_size, self.max_len-time-1), dtype=tf.int32)],2)
			print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: test_step: past_symbols:', past_symbols.get_shape().as_list())
			past_symbols = tf.reshape(past_symbols, [1,self.beam_size,self.max_len]) # shape [1, beam_size, caplen]
			print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: test_step: past_symbols:', past_symbols.get_shape().as_list())

			cond1 = tf.equal(symbols,tf.ones_like(symbols,tf.int32)*self.done_token) # condition on done sentence: shape [1, beam_size], bool
			#print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: test_step: cond1:', cond1.get_shape().as_list(), cond1)

			for_finished_logprobs = tf.where(cond1,past_logprobs,tf.ones_like(past_logprobs,tf.float32)* -1e5) # shape [1, beam_size]
			#print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: test_step: for_finished_logprobs:', for_finished_logprobs)

			done_indice_max = tf.cast(tf.argmax(for_finished_logprobs,axis=-1),tf.int32) # shape [1, ]
			logprobs_done_max = tf.reduce_max(for_finished_logprobs,reduction_indices=-1) # shape [1, ]
			#print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: test_step: done_indice_max:', done_indice_max, 'logprobs_done_max:', logprobs_done_max)

			done_past_symbols = tf.gather(tf.reshape(past_symbols,[self.beam_size,self.max_len]),done_indice_max) # shape [beam_size, caplen]
			logprobs_done_max = tf.div(-logprobs_done_max,tf.cast(time,tf.float32))
			cond2 = tf.greater(logprobs_finished_beams,logprobs_done_max)

			cond3 = tf.equal(done_past_symbols[:,time],self.done_token)
			cond4 = tf.equal(time,self.max_len-1)
			#print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: test_step: cond2:',cond2, 'cond3:',cond3, 'cond4:',cond4)
			finished_beams = tf.where(tf.logical_and(cond2,tf.logical_or(cond3,cond4)),
			                                done_past_symbols,
			                                finished_beams)
			logprobs_finished_beams = tf.where(tf.logical_and(cond2,tf.logical_or(cond3,cond4)),
											logprobs_done_max, 
											logprobs_finished_beams)

			print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: test_step: symbols:',symbols, 'h:',h)
			print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: test_step: past_symbols:',past_symbols, 'past_logprobs:',past_logprobs)
			print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: test_step: finished_beams:',finished_beams, 
				'logprobs_finished_beams:',logprobs_finished_beams)
			# symbols: shape [1, beam_size], beam words in t
			# h: shape [1, beam_size, output_dim]
			# past_symbols: shape [1, beam_size, caplen], beam words of time 1..t
			# past_logprobs: shape [1, beam_size]
			# finished_beams: best words of time 1..t
			# logprobs_finished_beams: shape [1,], cressponding prob of finished_beams
			return (time+1, symbols, h, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams)

		## here batch_size set to be 1
		captions = self.input_captions # shape [beam_batch, caplen]
		
		finished_beams = tf.zeros((self.batch_size, self.max_len), dtype=tf.int32) # shape [beam_batch, caplen]: record the final beam search result
		logprobs_finished_beams = tf.ones((self.batch_size,), dtype=tf.float32) * float('inf') # shape [beam_batch,] : the final prob
		
		x_0 = captions[:,0] # token <BOS>
		x_0 = tf.expand_dims(x_0,dim=-1) # shape [beam_batch, 1]
		x_0 = tf.tile(x_0,[1,self.beam_size]) # shape [beam_batch, beam_size]
		
		h_0 = tf.expand_dims(initial_state,dim=1) # shape [beam_batch, 1, output_dim]
		h_0 = tf.reshape(tf.tile(h_0,[1,self.beam_size,1]),[self.batch_size*self.beam_size,self.output_dim]) # shape [beam_batch*beam_size, output_dim]
		
		symbols, h, past_symbols, past_logprobs = take_step_zero(x_0, h_0)
		time = tf.constant(1, dtype='int32', name='time')
		timesteps = self.max_len
		## symbols: shape [1, beam_size], beam word index at t-1
		## h: shape [beam_size, output_dim], beam hidden states of decoder at t-1
		## past_symbols: shape [1, beam_size, caplen], past_symbols[:,:,0:t] is valid, [:,:,t:] is zeros
		## past_logprobs: shape [1, beam_size], logprobs of [1:t], prob multiplication, log prob sum
		print('SamModel_ObjectV_att_NoShare_test.py: beamSearchDecoder: after take_step_zero, symbols:',symbols,
			'h:',h, 'past_symbols:',past_symbols, 'past_logprobs:',past_logprobs)
		
		test_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=test_step,
	            loop_vars=(time, symbols, h, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams), #x_t: sysmbols, h_tm1: h
	            parallel_iterations=32,
	            swap_memory=True)

		out_finished_beams = test_out[-2]
		out_logprobs_finished_beams = test_out[-1]
		out_past_symbols = test_out[-4]

		return   out_finished_beams, out_logprobs_finished_beams, out_past_symbols

	def decoder_onestep(self):
		x_t = self.dec_x_t
		h_tm1 = self.dec_h_tm1
		input_feature = self.v_encoder_output
		o_input_feature = self.o_encoder_output

		def step(x_t,h_tm1):
			# x_t: shape [beam_size, d_w2v]
			# h_tm1: shape [beam_size, output_dim]
			# here the input_feature indicates v_vlads, shape [1(beam_batch), timestep(encoder), redu_dim*centers_num]
			# here the o_input_feature indicates obj_vlads, shape [obj_num, 1(beam_batch), timestep(encoder), redu_dim*centers_num]
			
			this_input_feature = tf.tile(input_feature, [self.beam_size,1,1])
			o_this_input_feature = tf.tile(o_input_feature, [1,self.beam_size,1,1])
			
			ori_feature = tf.reshape(this_input_feature,(-1,self.reduction_dim*self.centers_num))
			#ori_feature: shape [beam_size*timestep(encoder), redu_dim*centers_num]
			o_ori_feature = tf.reshape(o_this_input_feature,(-1,self.reduction_dim*self.centers_num))
			#o_ori_feature: shape [obj_num*beam_size*timestep(encoder), redu_dim*centers_num]
			
			#########
			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.timesteps,self.attention_dim))
			# attend_wx: shape [beam_size, timestep(encoder), attention_dim]
			attend_uh_tm1 = tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1) # shape [beam_size, 1, attention]
			attend_e = tf.nn.tanh(tf.add(attend_wx,attend_uh_tm1)) # shape [beam_size, timestep, attention]

			o_attend_wx = tf.reshape(tf.nn.xw_plus_b(o_ori_feature, self.o_W_a, self.o_b_a),(-1,self.timesteps,self.attention_dim))
			# o_attend_wx: shape [obj_num*beam_size, timestep(encoder), attention_dim]
			o_attend_uh_tm1 = tf.expand_dims(tf.matmul(h_tm1, self.o_U_a),dim=1) # shape [beam_size, 1, attention]
			o_attend_uh_tm1 = tf.tile(o_attend_uh_tm1, [self.obj_num, 1, 1]) # shape [obj_num*beam_size, 1, attention]
			o_attend_e = tf.nn.tanh(tf.add(o_attend_wx,o_attend_uh_tm1))
			
			#########
			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W)
			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.timesteps,1)),dim=1)
			attend_e = tf.reshape(attend_e,(-1,self.timesteps,1))

			o_attend_e = tf.matmul(tf.reshape(o_attend_e,(-1,self.attention_dim)),self.o_W)
			o_attend_e = tf.nn.softmax(tf.reshape(o_attend_e,(-1,self.timesteps,1)),dim=1)
			o_attend_e = tf.reshape(o_attend_e,(-1,self.timesteps,1))
			
			#########
			attend_fea = tf.multiply(tf.reshape(this_input_feature,(-1,self.timesteps,self.reduction_dim*self.centers_num)) , attend_e) 
			# attend_fea: shape [beam_size, timestep(encoder), redu_dim*centers_num]
			attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1) # shape [beam_size, redu_dim*centers_num]

			o_attend_fea = tf.multiply(tf.reshape(o_this_input_feature,(-1,self.timesteps,self.reduction_dim*self.centers_num)) , o_attend_e) 
			# attend_fea: shape [obj_num*beam_size, timestep(encoder), redu_dim*centers_num]
			o_attend_fea = tf.reduce_sum(o_attend_fea,reduction_indices=1) # shape [obj_num*beam_size, redu_dim*centers_num]
			
			#########
			### for object attention
			o_attend_fea = tf.reshape(o_attend_fea, [self.obj_num, -1, self.reduction_dim*self.centers_num]) # shape [obj_num, beam_size, redu_dim*centers_num]
			o_attend_fea = tf.transpose(o_attend_fea, perm=[1, 0, 2]) # shape [beam_size, obj_num, redu_dim*centers_num]
			ori_o_attend_fea = tf.reshape(o_attend_fea, [-1, self.reduction_dim*self.centers_num]) # shape [beam_size*obj_num, redu_dim*centers_num]

			attend_wx_obj = tf.reshape(tf.nn.xw_plus_b(ori_o_attend_fea, self.W_a_obj, self.b_a_obj),(-1,self.obj_num,self.attention_dim))
			# attend_wx_obj: shape [beam_size, obj_num, attention_dim]
			attend_uh_tm1_obj = tf.expand_dims(tf.matmul(h_tm1, self.U_a_obj),dim=1) # shape [beam_size, 1, attention_dim]
			attend_e_obj = tf.nn.tanh(tf.add(attend_wx_obj, attend_uh_tm1_obj)) # # shape [beam_size, obj_num, attention_dim]

			attend_e_obj = tf.matmul(tf.reshape(attend_e_obj,(-1,self.attention_dim)),self.W_obj) # shape [beam_size*obj_num]
			attend_e_obj = tf.nn.softmax(tf.reshape(attend_e_obj,(-1,self.obj_num,1)),dim=1) # shape [beam_size, obj_num, 1]
			attend_e_obj = tf.reshape(attend_e_obj,(-1,self.obj_num,1)) # shape [beam_size, obj_num, 1]

			attend_fea_obj = tf.multiply(tf.reshape(o_attend_fea, (-1,self.obj_num,self.reduction_dim*self.centers_num)), attend_e_obj) 
			#attend_fea_obj: shape [beam_size, obj_num, redu_dim*centers_num]
			o_attend_fea = tf.reduce_sum(attend_fea_obj, reduction_indices=1) # shape [beam_size, redu_dim*centers_num]
			
			#########
			attend_fea = tf.add(tf.matmul(attend_fea,self.A),tf.matmul(o_attend_fea, self.o_A)) # shape [beam_size, 3*output_dim]


			attend_fea_r = attend_fea[:,0:self.output_dim]
			attend_fea_z = attend_fea[:,self.output_dim:2*self.output_dim]
			attend_fea_h = attend_fea[:,2*self.output_dim::]

			preprocess_x = tf.nn.xw_plus_b(x_t, self.W_d, self.b_d)

			preprocess_x_r = preprocess_x[:,0:self.output_dim]
			preprocess_x_z = preprocess_x[:,self.output_dim:2*self.output_dim]
			preprocess_x_h = preprocess_x[:,2*self.output_dim::]

			r = hard_sigmoid(preprocess_x_r+ tf.matmul(h_tm1, self.U_d_r) + attend_fea_r)
			z = hard_sigmoid(preprocess_x_z+ tf.matmul(h_tm1, self.U_d_z) + attend_fea_z)
			hh = tf.nn.tanh(preprocess_x_h+ tf.matmul(r*h_tm1, self.U_d_h) + attend_fea_h)
			
			h = (1-z)*hh + z*h_tm1

			return h

		def test_step(x_t, h_tm1):
			#s_t: symbols, shape [1, beam_size]
			#past_symbols: shape [1, beam_size, caplen]
			#past_logprobs: shape [1, beam_size]
			
			x_t = tf.gather(self.T_w2v,x_t)*tf.gather(self.T_mask,x_t)
			x_t = tf.reshape(x_t,[self.batch_size*self.beam_size,self.d_w2v])
			h = step(x_t,h_tm1)

			drop_h = h # no dropout
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

			logprobs = tf.nn.log_softmax(predict_score_t)
			logprobs = tf.reshape(logprobs, [1, self.beam_size, self.voc_size]) # shape [1(beam_batch), beam_size, voc_size]
			
			return h, logprobs

		h, logprobs = test_step(x_t, h_tm1)
		
		return h, logprobs

	def build_model(self):

		self.init_parameters()
		'''
		encoder_vlads = tf.TensorArray(
	            dtype=tf.float32,
	            size=self.obj_num,
	            tensor_array_name='encoder_vlads')
		for obj_i in range(self.obj_num):
			obj_vlad = self.encoder(self.obj_feature[obj_i]) # shape [batch, timestep, redu_dim*centers_num]
			encoder_vlads = encoder_vlads.write(obj_i, obj_vlad)
		if hasattr(encoder_vlads, 'stack'):
			encoder_output = encoder_vlads.stack()
		else:
			encoder_output = encoder_vlads.pack() #shape [obj_num, batch, timestep, redu_dim*centers_num]
		'''
		#input_feature = tf.expand_dims(self.video_feature, dim=0)
		#input_feature = tf.concat([input_feature, self.obj_feature],0) # shape [1+obj_num, batch, timestep, h, w, c]
		#print('SamModel_ObjectV_att_NoShare_test.py: build_model: input_feature:', input_feature.get_shape().as_list())
		#encoder_output = self.encoder_allobjs(input_feature) #shape [1+obj_num, batch, timestep, redu_dim*centers_num]

		v_encoder_output = self.encoder(self.video_feature) #shape [batch, timestep, redu_dim*centers_num]
		o_encoder_output = self.encoder_allobjs(self.obj_feature) #shape [obj_num, batch, timestep, redu_dim*centers_num]
		
		v_last_output = tf.reduce_mean(self.video_feature,axis=[1,2,3]) # shape [batch, c(2048)]
		o_last_output = tf.reduce_mean(self.obj_feature,axis=[0,2,3,4]) # shape [batch, c(2048)]

		if self.output_dim!=self.feature_dim:
			print '$$$$$apart$$$$   output_dim:', self.output_dim,' self.feature_dim:', self.feature_dim #new
			v_last_output = tf.nn.xw_plus_b(v_last_output, self.liner_W, self.liner_b) # shape [batch, output_dim]
			o_last_output = tf.nn.xw_plus_b(o_last_output, self.o_liner_W, self.o_liner_b) # shape [batch, output_dim]
			last_output = tf.add(v_last_output, o_last_output) # shape [batch, output_dim]
		
		print('SamModel_ObjectV_att_NoShare_test.py: build_model: after encoder, last_output:', last_output.get_shape().as_list())
		print('SamModel_ObjectV_att_NoShare_test.py: build_model: v_encoder_output:', v_encoder_output.get_shape().as_list(),
			'o_encoder_output (VLAD):', o_encoder_output.get_shape().as_list())

		predict_score, predict_words , loss_mask= self.decoder_objV_att(last_output, v_encoder_output, o_encoder_output)
		#predict_score: shape [batch, caplen, voc_size], scores for train_hidden_states
		#predict_words: shape [batch, caplen], for test hidden_states
		#loss_mask: shape [batch, caplen], float32, the float version of mask (bool)
		print('SamModel_ObjectV_att_NoShare_test.py: build_model: after decoder, predict_score:', predict_score.get_shape().as_list(),
			'predict_words:', predict_words.get_shape().as_list(), 'loss_mask:', loss_mask.get_shape().as_list())
		
		#last_output: shape [batch, output_dim], here the batch_size must be 1; encoder_output: shape [batch, timestep, redu_dim*centers_num]
		finished_beam, logprobs_finished_beams, past_symbols = self.beamSearchDecoder_objV_att(last_output, v_encoder_output, o_encoder_output)
		print('SamModel_ObjectV_att_NoShare_test.py: build_model: after beamSearchDecoder, finished_beam:', finished_beam.get_shape().as_list(),
			'logprobs_finished_beams:', logprobs_finished_beams.get_shape().as_list(), 'past_symbols:', past_symbols.get_shape().as_list())
		#predict_score: shape [batch, caplen, voc_size], scores for train_hidden_states
		#loss_mask: shape [batch, caplen], float32, the float version of mask (bool)
		#finished_beam: shape [1, caplen], the final symbols of beamsearchDecoder
		#logprobs_finished_beams: shape [1, ], prob of finished_beam
		#past_symbols: shape [1, beam_size, caplen]
		return predict_score, loss_mask, finished_beam, logprobs_finished_beams, past_symbols

	def build_model_test(self):

		self.init_parameters()
		
		v_encoder_output = self.encoder(self.video_feature) #shape [batch, timestep, redu_dim*centers_num]
		o_encoder_output = self.encoder_allobjs(self.obj_feature) #shape [obj_num, batch, timestep, redu_dim*centers_num]
		
		v_last_output = tf.reduce_mean(self.video_feature,axis=[1,2,3]) # shape [batch, c(2048)]
		o_last_output = tf.reduce_mean(self.obj_feature,axis=[0,2,3,4]) # shape [batch, c(2048)]

		if self.output_dim!=self.feature_dim:
			v_last_output = tf.nn.xw_plus_b(v_last_output, self.liner_W, self.liner_b) # shape [batch, output_dim]
			o_last_output = tf.nn.xw_plus_b(o_last_output, self.o_liner_W, self.o_liner_b) # shape [batch, output_dim]
			last_output = tf.add(v_last_output, o_last_output) # shape [batch, output_dim]
		
		dec_h, dec_logprobs = self.decoder_onestep() 

		return (last_output, v_encoder_output, o_encoder_output), (dec_h, dec_logprobs)