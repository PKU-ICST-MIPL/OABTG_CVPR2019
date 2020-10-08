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
	'''
	def __init__(self, input_feature, input_captions, voc_size, d_w2v, output_dim, 
		reduction_dim=512, centers_num=16, filter_size=1, stride=[1,1,1,1], pad='SAME', 
		done_token=3, max_len = 20, beamsearch_batchsize = 1, beam_size=5,
		attention_dim = 100, dropout=0.5,inner_activation='hard_sigmoid',
		activation='tanh', return_sequences=True, bottleneck=256):'''
	def __init__(self, input_feature, input_captions, dec_x_t, dec_h_tm1, f_enc_out, b_enc_out, voc_size, d_w2v, output_dim, 
		reduction_dim=512, centers_num=16, filter_size=1, stride=[1,1,1,1], pad='SAME', 
		done_token=3, max_len = 20, beamsearch_batchsize = 1, beam_size=5,
		attention_dim = 100, dropout=0.5,inner_activation='hard_sigmoid',
		activation='tanh', return_sequences=True, bottleneck=256):

		self.reduction_dim=reduction_dim
		
		self.input_feature = tf.transpose(input_feature,perm=[0,1,3,4,2]) # after transpose teh shape should be (batch, timesteps, height, width, channels)

		self.input_captions = input_captions

		self.dec_x_t = dec_x_t
		self.dec_h_tm1 = dec_h_tm1
		self.f_encoder_output = f_enc_out
		self.b_encoder_output = b_enc_out

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



		self.enc_in_shape = self.input_feature.get_shape().as_list()
		self.decoder_input_shape = self.input_captions.get_shape().as_list()
		## edited by zjc
		print('SamModel_test.py: __init__: self.enc_in_shape:', self.enc_in_shape, 'self.decoder_input_shape:', self.decoder_input_shape)
		self.bottleneck = bottleneck

	def init_parameters(self):
		print('init_parameters ...')

		self.redu_W = tf.get_variable("redu_W", shape=[1, 1, self.enc_in_shape[-1], self.reduction_dim], 
										initializer=tf.contrib.layers.xavier_initializer())
		self.redu_b = tf.get_variable("redu_b",initializer=tf.random_normal([self.reduction_dim],stddev=1./math.sqrt(self.reduction_dim)))

		self.W_e = tf.get_variable("W_e", shape=[3, 3, self.reduction_dim, 3*self.centers_num], initializer=tf.contrib.layers.xavier_initializer())
		self.b_e = tf.get_variable("b_e",initializer=tf.random_normal([3*self.centers_num],stddev=1./math.sqrt(3*self.centers_num)))

		self.f_centers = tf.get_variable("f_centers",[1, 1, 1, self.reduction_dim, self.centers_num],
			initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))

		encoder_h2h_shape = (self.filter_size, self.filter_size, self.centers_num, self.centers_num)
		self.U_e_r = tf.get_variable("U_e_r", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_z = tf.get_variable("U_e_z", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer())
		self.U_e_h = tf.get_variable("U_e_h", shape=encoder_h2h_shape, initializer=tf.contrib.layers.xavier_initializer()) 

		if self.output_dim!=self.enc_in_shape[-1]:
			print "$$$init$$$       output_dim:",self.output_dim,' enc_in_shape:',self.enc_in_shape[-1]#new
			print('the dimension of input feature != hidden size')
			self.liner_W = tf.get_variable("liner_W",[self.enc_in_shape[-1], self.output_dim],
				initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.enc_in_shape[-1])))

			self.liner_b = tf.get_variable("liner_b",initializer=tf.random_normal([self.output_dim],stddev=1./math.sqrt(self.output_dim)))

		# decoder parameters
		self.T_w2v, self.T_mask = self.init_embedding_matrix()

		decoder_i2h_shape = (self.d_w2v,3*self.output_dim)
		decoder_h2h_shape = (self.output_dim,self.output_dim)

		self.W_d = tf.get_variable("W_d",decoder_i2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.d_w2v)))
		self.b_d = tf.get_variable("b_d",initializer = tf.random_normal([3*self.output_dim], stddev=1./math.sqrt(3*self.output_dim)))
		
		self.U_d_r = tf.get_variable("U_d_r",decoder_h2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
		self.U_d_z = tf.get_variable("U_d_z",decoder_h2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
		self.U_d_h = tf.get_variable("U_d_h",decoder_h2h_shape,initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))

		# attention parameters
		self.W_a = tf.get_variable("W_a",[self.reduction_dim*self.centers_num,self.attention_dim],
			initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.reduction_dim*self.centers_num)))

		self.U_a = tf.get_variable("U_a",[self.output_dim,self.attention_dim],initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
		self.b_a = tf.get_variable("b_a",initializer = tf.random_normal([self.attention_dim],stddev=1. / math.sqrt(self.attention_dim)))

		self.W = tf.get_variable("W",(self.attention_dim,1),initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.attention_dim)))

		self.A = tf.get_variable("A",(self.reduction_dim*self.centers_num,3*self.output_dim),
			initializer=tf.random_normal_initializer(stddev=1./ math.sqrt(self.reduction_dim*self.centers_num)))


		self.b_W_a = tf.get_variable("b_W_a",[self.reduction_dim*self.centers_num,self.attention_dim],
			initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.reduction_dim*self.centers_num)))
		self.b_U_a = tf.get_variable("b_U_a",[self.output_dim,self.attention_dim],initializer=tf.random_normal_initializer(stddev=1./math.sqrt(self.output_dim)))
		self.b_b_a = tf.get_variable("b_b_a",initializer = tf.random_normal([self.attention_dim],stddev=1. / math.sqrt(self.attention_dim)))
		self.b_W = tf.get_variable("b_W",(self.attention_dim,1),initializer=tf.random_normal_initializer(stddev=1. / math.sqrt(self.attention_dim)))

		self.b_A = tf.get_variable("b_A",(self.reduction_dim*self.centers_num,3*self.output_dim),
			initializer=tf.random_normal_initializer(stddev=1./ math.sqrt(self.reduction_dim*self.centers_num)))

		# classification parameters
		self.W_c = tf.get_variable("W_c",[self.output_dim,self.voc_size],
			initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(self.output_dim)))
		self.b_c = tf.get_variable("b_c",initializer = tf.random_normal([self.voc_size],stddev=1./math.sqrt(self.voc_size)))


		self.down_W = tf.get_variable("down_W", shape=[1, 1, self.enc_in_shape[-1], self.bottleneck], 
										initializer=tf.contrib.layers.xavier_initializer())
		self.down_b = tf.get_variable("down_b",initializer=tf.random_normal([self.bottleneck],stddev=1./math.sqrt(self.bottleneck)))
		
		self.keep_W = tf.get_variable("keep_W", shape=[3, 3, self.bottleneck, self.bottleneck], 
										initializer=tf.contrib.layers.xavier_initializer())
		self.keep_b = tf.get_variable("keep_b",initializer=tf.random_normal([self.bottleneck],stddev=1./math.sqrt(self.bottleneck)))

		self.up_W = tf.get_variable("up_W", shape=[1, 1, self.bottleneck, self.enc_in_shape[-1]], 
										initializer=tf.contrib.layers.xavier_initializer())
		self.up_b = tf.get_variable("up_b",initializer=tf.random_normal([self.enc_in_shape[-1]],stddev=1./math.sqrt(self.enc_in_shape[-1])))


	def init_embedding_matrix(self):

		'''init word embedding matrix
		'''
		print('SamModel_test.py: init_embedding_matrix starts ...')
		voc_size = self.voc_size
		d_w2v = self.d_w2v	
		print('SamModel_test.py: init_embedding_matrix: voc_size=', voc_size, 'd_w2v=', d_w2v)

		np_mask = np.vstack((np.zeros(d_w2v),np.ones((voc_size-1,d_w2v))))
		T_mask = tf.constant(np_mask, tf.float32, name='LUT_mask')
		print('SamModel_test.py: init_embedding_matrix: T_mask:',T_mask.get_shape().as_list())

		LUT = np.zeros((voc_size, d_w2v), dtype='float32')
		for v in range(voc_size):
			LUT[v] = rng.randn(d_w2v)
			LUT[v] = LUT[v] / (np.linalg.norm(LUT[v]) + 1e-6)

		# word 0 is blanked out, word 1 is 'UNK'
		LUT[0] = np.zeros((d_w2v))
		# setup LUT!
		T_w2v = tf.Variable(LUT.astype('float32'),trainable=True)
		print('SamModel_test.py: init_embedding_matrix: T_w2v:',T_w2v.get_shape().as_list())
		print('SamModel_test.py: init_embedding_matrix ends ...')
		return T_w2v, T_mask 

	def encoder(self):
		print('down building encoder ... ...')
		timesteps = self.enc_in_shape[1]
		# reduction
		input_feature = self.input_feature
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.enc_in_shape[4]])
		t_ori_feature = input_feature

		input_feature = tf.add(tf.nn.conv2d(input_feature, self.redu_W, self.stride, self.pad, name='reduction_wx'),
			tf.reshape(self.redu_b,[1, 1, 1, self.reduction_dim]))

		t_feature = tf.add(tf.nn.conv2d(t_ori_feature, self.down_W, self.stride, self.pad, name='down_sampling'),
			tf.reshape(self.down_b,[1, 1, 1, self.bottleneck]))
		t_feature = tf.nn.relu(t_feature)
		t_feature = tf.add(tf.nn.conv2d(t_feature, self.keep_W, self.stride, self.pad, name='keep_sampling'),
			tf.reshape(self.keep_b,[1, 1, 1, self.bottleneck]))
		t_feature = tf.nn.relu(t_feature)
		t_feature = tf.add(tf.nn.conv2d(t_feature, self.up_W, self.stride, self.pad, name='up_sampling'),
			tf.reshape(self.up_b,[1, 1, 1, self.enc_in_shape[-1]]))
		t_feature = tf.nn.relu(t_ori_feature+t_feature)

		threshold = tf.reduce_mean(t_feature,axis=[-1],keep_dims=True)
		print('threshold.get_shape',threshold.get_shape().as_list())
		
		max_value = tf.reduce_max(threshold,axis=[1,2],keep_dims=True)
		threshold = threshold/max_value
		threshold =  tf.reshape(tf.tile(threshold,[1,1,1,self.centers_num]),
			[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.centers_num])
		
		input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[1],self.enc_in_shape[2],
			self.enc_in_shape[3],self.reduction_dim])
		input_feature = tf.nn.relu(input_feature)
		self.enc_in_shape = input_feature.get_shape().as_list()
		assignment = tf.reshape(input_feature,[-1,self.enc_in_shape[2],self.enc_in_shape[3],self.reduction_dim])
		assignment = tf.add(tf.nn.conv2d(assignment, self.W_e, self.stride, self.pad, name='w_conv_x'),tf.reshape(self.b_e,[1, 1, 1, 3*self.centers_num]))
		assignment = tf.reshape(assignment,[-1,self.enc_in_shape[1],self.enc_in_shape[2],self.enc_in_shape[3],3*self.centers_num])

		axis = [1,0]+list(range(2,5)) 
		assignment = tf.transpose(assignment, perm=axis)

		input_assignment = tf.TensorArray(
	            dtype=assignment.dtype,
	            size=timesteps,
	            tensor_array_name='input_assignment')
		if hasattr(input_assignment, 'unstack'):
			input_assignment = input_assignment.unstack(assignment)
		else:
			input_assignment = input_assignment.unpack(assignment)	

		hidden_states = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='hidden_states')

		def get_init_state(x, output_dims):
			initial_state = tf.zeros_like(x)
			initial_state = tf.reduce_sum(initial_state,axis=[1,4])
			initial_state = tf.expand_dims(initial_state,dim=-1)
			initial_state = tf.tile(initial_state,[1,1,1,output_dims])
			return initial_state
		def step(time, hidden_states, h_tm1):
			assign_t = input_assignment.read(time)

			assign_t_r, assign_t_z, assign_t_h = tf.split(assign_t,3,axis=3)
			
			r = hard_sigmoid(assign_t_r+ tf.nn.conv2d(h_tm1, self.U_e_r, self.stride, self.pad, name='r'))
			z = hard_sigmoid(assign_t_z+ tf.nn.conv2d(h_tm1, self.U_e_z, self.stride, self.pad, name='z'))

			hh = tf.tanh(assign_t_h+ tf.nn.conv2d(r*h_tm1, self.U_e_h, self.stride, self.pad, name='uh_hh'))

			h = (1-z)*hh + z*h_tm1
			
			hidden_states = hidden_states.write(time, h)

			return (time+1,hidden_states, h)

		time = tf.constant(0, dtype='int32', name='time')
		initial_state = get_init_state(input_feature,self.centers_num)

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
			assignment = hidden_states.pack()

		axis = [1,0]+list(range(2,5)) 
		assignment = tf.transpose(assignment, perm=axis)

		assignment = tf.reshape(assignment,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.centers_num])

		# backgroung,front
		f_assignment = threshold*assignment
		b_assignment = (1-threshold)*assignment

		def apart(assignment,input_feature,centers):
			a_sum = tf.reduce_sum(assignment,-2,keep_dims=True)
			a = tf.multiply(a_sum,centers) 
			assignment = tf.transpose(assignment,perm=[0,2,1])

			input_feature = tf.reshape(input_feature,[-1,self.enc_in_shape[2]*self.enc_in_shape[3],self.reduction_dim])
			vlad = tf.matmul(assignment,input_feature) 
			vlad = tf.transpose(vlad, perm=[0,2,1])
			tf.summary.histogram('vlad',vlad)
			
			# for differnce
			vlad = tf.subtract(vlad,a)

			vlad = tf.reshape(vlad,[-1,self.enc_in_shape[-1],self.centers_num])
			vlad = tf.nn.l2_normalize(vlad,1)

			vlad = tf.reshape(vlad,[-1,self.enc_in_shape[1],self.enc_in_shape[-1]*self.centers_num])
			vlad = tf.nn.l2_normalize(vlad,2)
			
			return vlad

		f_vlad = apart(f_assignment,input_feature,self.f_centers)
		b_vlad = apart(b_assignment,input_feature,self.f_centers)

		last_output = tf.reduce_mean(self.input_feature,axis=[1,2,3])
		if self.output_dim!=self.input_feature.get_shape().as_list()[-1]:
			print '$$$$$apart$$$$   output_dim:', self.output_dim,' input_feature:', self.input_feature.get_shape().as_list()[-1] #new

                        print('the dimension of input feature != hidden size')
			last_output = tf.nn.xw_plus_b(last_output,self.liner_W, self.liner_b)

		return last_output, f_vlad, b_vlad

	def decoder(self, initial_state, input_feature, b_input_feature):
	
		#captions: (batch_size x timesteps) ,int32
		#d_w2v: dimension of word 2 vector
	
		print('up self.dropout',self.dropout)
		captions = self.input_captions

		print('up building decoder ... ...')
		mask =  tf.not_equal(captions,0)

		loss_mask = tf.cast(mask,tf.float32)

		embedded_captions = tf.gather(self.T_w2v,captions)*tf.gather(self.T_mask,captions)

		timesteps = self.decoder_input_shape[1]

		axis = [1,0]+list(range(2,3)) 
		embedded_captions = tf.transpose(embedded_captions, perm=axis) 

		input_embedded_words = tf.TensorArray(
	            dtype=embedded_captions.dtype,
	            size=timesteps,
	            tensor_array_name='input_embedded_words')

		if hasattr(input_embedded_words, 'unstack'):
			input_embedded_words = input_embedded_words.unstack(embedded_captions)
		else:
			input_embedded_words = input_embedded_words.unpack(embedded_captions)	

		# preprocess mask
		mask = tf.expand_dims(mask,dim=-1)
		
		mask = tf.transpose(mask,perm=axis)

		input_mask = tf.TensorArray(
			dtype=mask.dtype,
			size=timesteps,
			tensor_array_name='input_mask'
			)

		if hasattr(input_mask, 'unstack'):
			input_mask = input_mask.unstack(mask)
		else:
			input_mask = input_mask.unpack(mask)


		train_hidden_state = tf.TensorArray(
	            dtype=tf.float32,
	            size=timesteps,
	            tensor_array_name='train_hidden_state')

		def step(x_t,h_tm1):

			ori_feature = tf.reshape(input_feature,(-1,self.enc_in_shape[-1]*self.centers_num))
			b_ori_feature = tf.reshape(b_input_feature,(-1,self.enc_in_shape[-1]*self.centers_num))
			########
			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.enc_in_shape[1],self.attention_dim))
			attend_uh_tm1 = tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1)
			attend_e = tf.nn.tanh(tf.add(attend_wx,attend_uh_tm1))

			b_attend_wx = tf.reshape(tf.nn.xw_plus_b(b_ori_feature, self.b_W_a, self.b_b_a),(-1,self.enc_in_shape[1],self.attention_dim))
			b_attend_uh_tm1 = tf.expand_dims(tf.matmul(h_tm1, self.b_U_a),dim=1)
			b_attend_e = tf.nn.tanh(tf.add(b_attend_wx,b_attend_uh_tm1))

			########
			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W) # batch_size * timestep
			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.enc_in_shape[1],1)),dim=1)
			attend_e = tf.reshape(attend_e,(-1,self.enc_in_shape[1],1))


			b_attend_e = tf.matmul(tf.reshape(b_attend_e,(-1,self.attention_dim)),self.b_W) # batch_size * timestep
			b_attend_e = tf.nn.softmax(tf.reshape(b_attend_e,(-1,self.enc_in_shape[1],1)),dim=1)
			b_attend_e = tf.reshape(b_attend_e,(-1,self.enc_in_shape[1],1))

			attend_fea = tf.multiply(input_feature , attend_e)
			attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)

			b_attend_fea = tf.multiply(b_input_feature , b_attend_e)
			b_attend_fea = tf.reduce_sum(b_attend_fea,reduction_indices=1)

			attend_fea = tf.add(tf.matmul(attend_fea,self.A),tf.matmul(b_attend_fea,self.b_A))

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

		def train_step(time, train_hidden_state, h_tm1):
			x_t = input_embedded_words.read(time) # batch_size * dim
			mask_t = input_mask.read(time)

			h = step(x_t,h_tm1)

			tiled_mask_t = tf.tile(mask_t, tf.stack([1, h.get_shape().as_list()[1]]))

			h = tf.where(tiled_mask_t, h, h_tm1) # (batch_size, output_dims)
			
			train_hidden_state = train_hidden_state.write(time, h)

			return (time+1,train_hidden_state,h)

		time = tf.constant(0, dtype='int32', name='time')

		train_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=train_step,
	            loop_vars=(time, train_hidden_state, initial_state),
	            parallel_iterations=32,
	            swap_memory=True)

		train_hidden_state = train_out[1]
		train_last_output = train_out[-1] 
		
		if hasattr(train_hidden_state, 'stack'):
			train_outputs = train_hidden_state.stack()
		else:
			train_outputs = train_hidden_state.pack()

		axis = [1,0] + list(range(2,3))
		train_outputs = tf.transpose(train_outputs,perm=axis)

		train_outputs = tf.reshape(train_outputs,(-1,self.output_dim))
		train_outputs = tf.nn.dropout(train_outputs, self.dropout)
		predict_score = tf.matmul(train_outputs,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))
		predict_score = tf.reshape(predict_score,(-1,timesteps,self.voc_size))

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
			x_t = test_input_embedded_words.read(time) # batch_size * dim

			h = step(x_t,h_tm1)

			test_hidden_state = test_hidden_state.write(time, h)

			drop_h = h*self.dropout
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

			predict_score_t = tf.nn.softmax(predict_score_t,dim=-1)
			predict_word_t = tf.argmax(predict_score_t,-1)

			predict_words = predict_words.write(time, predict_word_t) # output

			predict_word_t = tf.gather(self.T_w2v,predict_word_t)*tf.gather(self.T_mask,predict_word_t)

			test_input_embedded_words = test_input_embedded_words.write(time+1,predict_word_t)

			return (time+1,test_hidden_state, test_input_embedded_words, predict_words, h)

		time = tf.constant(0, dtype='int32', name='time')

		test_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=test_step,
	            loop_vars=(time, test_hidden_state, test_input_embedded_words, predict_words, initial_state),
	            parallel_iterations=32,
	            swap_memory=True)

		predict_words = test_out[-2]
		
		if hasattr(predict_words, 'stack'):
			predict_words = predict_words.stack()
		else:
			predict_words = predict_words.pack()

		axis = [1,0] + list(range(2,3))

		predict_words = tf.transpose(predict_words,perm=[1,0])
		predict_words = tf.reshape(predict_words,(-1,timesteps))

		return predict_score, predict_words, loss_mask

	def beamSearchDecoder(self, initial_state, input_feature, b_input_feature):
		
		def step(x_t,h_tm1):

			ori_feature = tf.reshape(input_feature,(-1,self.enc_in_shape[-1]*self.centers_num))
			b_ori_feature = tf.reshape(b_input_feature,(-1,self.enc_in_shape[-1]*self.centers_num))

			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.enc_in_shape[1],self.attention_dim))
			attend_uh_tm1 = tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1)
			attend_e = tf.nn.tanh(tf.add(attend_wx,attend_uh_tm1))

			b_attend_wx = tf.reshape(tf.nn.xw_plus_b(b_ori_feature, self.b_W_a, self.b_b_a),(-1,self.enc_in_shape[1],self.attention_dim))
			b_attend_uh_tm1 = tf.expand_dims(tf.matmul(h_tm1, self.b_U_a),dim=1)
			b_attend_e = tf.nn.tanh(tf.add(b_attend_wx,b_attend_uh_tm1))

			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W) # batch_size * timestep
			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.enc_in_shape[1],1)),dim=1)
			attend_e = tf.reshape(attend_e,(-1,self.enc_in_shape[1],1))


			b_attend_e = tf.matmul(tf.reshape(b_attend_e,(-1,self.attention_dim)),self.b_W) # batch_size * timestep
			b_attend_e = tf.nn.softmax(tf.reshape(b_attend_e,(-1,self.enc_in_shape[1],1)),dim=1)
			b_attend_e = tf.reshape(b_attend_e,(-1,self.enc_in_shape[1],1))

			attend_fea = tf.multiply(input_feature, attend_e)
			attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)

			b_attend_fea = tf.multiply(b_input_feature, b_attend_e)
			b_attend_fea = tf.reduce_sum(b_attend_fea,reduction_indices=1)
		
			attend_fea = tf.add(tf.matmul(attend_fea,self.A),tf.matmul(b_attend_fea,self.b_A))

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
			## zjc
			print('SamModel_test.py: beamSearchDecoder: take_step_zero starts ...')
			x_0 = tf.gather(self.T_w2v,x_0)*tf.gather(self.T_mask,x_0)
			print('SamModel_test.py: beamSearchDecoder: take_step_zero: x_0:', x_0.get_shape().as_list())
			x_0 = tf.reshape(x_0,[self.batch_size*self.beam_size,self.d_w2v])
			print('SamModel_test.py: beamSearchDecoder: take_step_zero: x_0:', x_0.get_shape().as_list())
			
			h = step(x_0,h_0)
			print('SamModel_test.py: beamSearchDecoder: take_step_zero: h:', h.get_shape().as_list())
			
			drop_h = h
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))
			logprobs = tf.nn.log_softmax(predict_score_t)

			print('logrobs.get_shape().as_list():',logprobs.get_shape().as_list())

			logprobs_batched = tf.reshape(logprobs, [-1, self.beam_size, self.voc_size])
			print('SamModel_test.py: beamSearchDecoder: take_step_zero: logprobs_batched:', logprobs_batched.get_shape().as_list())
			
			print('SamModel_test.py: beamSearchDecoder: take_step_zero: logprobs_batched[:,0,:]:', logprobs_batched[:,0,:].get_shape().as_list())
			past_logprobs, indices = tf.nn.top_k(
			        logprobs_batched[:,0,:],self.beam_size)
			print('SamModel_test.py: beamSearchDecoder: take_step_zero: past_logprobs:', past_logprobs.get_shape().as_list(),
				'indices:', indices.get_shape().as_list())
			symbols = indices % self.voc_size
			parent_refs = indices//self.voc_size
			h = tf.gather(h,  tf.reshape(parent_refs,[-1]))
			print('symbols.shape',symbols.get_shape().as_list())

			past_symbols = tf.concat([tf.expand_dims(symbols, 2), tf.zeros((self.batch_size, self.beam_size, self.max_len-1), dtype=tf.int32)],-1)
			return symbols, h, past_symbols, past_logprobs


		def test_step(time, x_t, h_tm1, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams):

			x_t = tf.gather(self.T_w2v,x_t)*tf.gather(self.T_mask,x_t)
			x_t = tf.reshape(x_t,[self.batch_size*self.beam_size,self.d_w2v])
			h = step(x_t,h_tm1)

			print('h.shape()',h.get_shape().as_list())
			drop_h = h
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

			logprobs = tf.nn.log_softmax(predict_score_t)
			logprobs = tf.reshape(logprobs, [1, self.beam_size, self.voc_size])
			print('SamModel_test.py: beamSearchDecoder: test_step: logprobs:', logprobs.get_shape().as_list(), 
				'past_logprobs:', past_logprobs.get_shape().as_list())
		
			logprobs = logprobs+tf.expand_dims(past_logprobs, 2)
			past_logprobs, topk_indices = tf.nn.top_k(
			    tf.reshape(logprobs, [1, self.beam_size * self.voc_size]),
			    self.beam_size, 
			    sorted=False
			)       
			print('SamModel_test.py: beamSearchDecoder: test_step: past_logprobs:', past_logprobs.get_shape().as_list(),
				'topk_indices:', topk_indices.get_shape().as_list())
			symbols = topk_indices % self.voc_size
			symbols = tf.reshape(symbols, [1,self.beam_size])
			parent_refs = topk_indices // self.voc_size

			h = tf.gather(h,  tf.reshape(parent_refs,[-1]))
			past_symbols_batch_major = tf.reshape(past_symbols[:,:,0:time], [-1, time])

			beam_past_symbols = tf.gather(past_symbols_batch_major,  parent_refs)
			print('SamModel_test.py: beamSearchDecoder: test_step: beam_past_symbols:', beam_past_symbols.get_shape().as_list())
			past_symbols = tf.concat([beam_past_symbols, tf.expand_dims(symbols, 2), tf.zeros((1, self.beam_size, self.max_len-time-1), dtype=tf.int32)],2)
			print('SamModel_test.py: beamSearchDecoder: test_step: past_symbols:', past_symbols.get_shape().as_list())
			past_symbols = tf.reshape(past_symbols, [1,self.beam_size,self.max_len])
			print('SamModel_test.py: beamSearchDecoder: test_step: past_symbols:', past_symbols.get_shape().as_list())

			cond1 = tf.equal(symbols,tf.ones_like(symbols,tf.int32)*self.done_token) # condition on done sentence
			print('SamModel_test.py: beamSearchDecoder: test_step: cond1:', cond1.get_shape().as_list(), cond1)

			for_finished_logprobs = tf.where(cond1,past_logprobs,tf.ones_like(past_logprobs,tf.float32)* -1e5)
			print('SamModel_test.py: beamSearchDecoder: test_step: for_finished_logprobs:', for_finished_logprobs)

			done_indice_max = tf.cast(tf.argmax(for_finished_logprobs,axis=-1),tf.int32)
			logprobs_done_max = tf.reduce_max(for_finished_logprobs,reduction_indices=-1)
			print('SamModel_test.py: beamSearchDecoder: test_step: done_indice_max:', done_indice_max, 'logprobs_done_max:', logprobs_done_max)

			done_past_symbols = tf.gather(tf.reshape(past_symbols,[self.beam_size,self.max_len]),done_indice_max)
			logprobs_done_max = tf.div(-logprobs_done_max,tf.cast(time,tf.float32))
			cond2 = tf.greater(logprobs_finished_beams,logprobs_done_max)

			cond3 = tf.equal(done_past_symbols[:,time],self.done_token)
			cond4 = tf.equal(time,self.max_len-1)
			print('SamModel_test.py: beamSearchDecoder: test_step: cond2:',cond2, 'cond3:',cond3, 'cond4:',cond4)
			finished_beams = tf.where(tf.logical_and(cond2,tf.logical_or(cond3,cond4)),
			                                done_past_symbols,
			                                finished_beams)
			logprobs_finished_beams = tf.where(tf.logical_and(cond2,tf.logical_or(cond3,cond4)),
											logprobs_done_max, 
											logprobs_finished_beams)

			print('SamModel_test.py: beamSearchDecoder: test_step: symbols:',symbols, 'h:',h)
			print('SamModel_test.py: beamSearchDecoder: test_step: past_symbols:',past_symbols, 'past_logprobs:',past_logprobs)
			print('SamModel_test.py: beamSearchDecoder: test_step: finished_beams:',finished_beams, 
				'logprobs_finished_beams:',logprobs_finished_beams)

			return (time+1, symbols, h, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams)


		captions = self.input_captions
		## zjc
		print('SamModel_test.py: beamSearchDecoder: captions:', captions.get_shape().as_list())
		finished_beams = tf.zeros((self.batch_size, self.max_len), dtype=tf.int32)
		logprobs_finished_beams = tf.ones((self.batch_size,), dtype=tf.float32) * float('inf')
		print('SamModel_test.py: beamSearchDecoder: finished_beams:', finished_beams.get_shape().as_list(), 
			'logprobs_finished_beams:', logprobs_finished_beams.get_shape().as_list())
		
		x_0 = captions[:,0]
		x_0 = tf.expand_dims(x_0,dim=-1)
		print('x_0',x_0.get_shape().as_list())
		x_0 = tf.tile(x_0,[1,self.beam_size])
		print('SamModel_test.py: beamSearchDecoder: after tile, x_0',x_0.get_shape().as_list())

		h_0 = tf.expand_dims(initial_state,dim=1)
		print('SamModel_test.py: beamSearchDecoder: h_0',h_0.get_shape().as_list())
		h_0 = tf.reshape(tf.tile(h_0,[1,self.beam_size,1]),[self.batch_size*self.beam_size,self.output_dim])
		print('SamModel_test.py: beamSearchDecoder: after tile, h_0',h_0.get_shape().as_list())

		symbols, h, past_symbols, past_logprobs = take_step_zero(x_0, h_0)
		time = tf.constant(1, dtype='int32', name='time')
		timesteps = self.max_len
		## symbols: shape [1, 5], beam word index at t-1
		## h: shape [5, 512], beam hidden states of decoder at t-1
		## past_symbols: shape [1, 5, max_len(17)], past_symbols[:,:,0:time] is valid, [:,:,time:] is zeros
		## past_logprobs: shape [1, 5], logprobs of [1:t], prob multiplication, log prob sum
		print('SamModel_test.py: beamSearchDecoder: after take_step_zero, symbols:',symbols,
			'h:',h, 'past_symbols:',past_symbols, 'past_logprobs:',past_logprobs)
		
		test_out = tf.while_loop(
	            cond=lambda time, *_: time < timesteps,
	            body=test_step,
	            loop_vars=(time, symbols, h, past_symbols, past_logprobs, finished_beams, logprobs_finished_beams),
	            parallel_iterations=32,
	            swap_memory=True)

		out_finished_beams = test_out[-2]
		out_logprobs_finished_beams = test_out[-1]
		out_past_symbols = test_out[-4]

		return   out_finished_beams, out_logprobs_finished_beams, out_past_symbols

	def decoder_onestep(self):
		
		x_t = self.dec_x_t
		h_tm1 = self.dec_h_tm1
		input_feature = self.f_encoder_output
		b_input_feature = self.b_encoder_output

		def step(x_t,h_tm1):

			ori_feature = tf.reshape(input_feature,(-1,self.enc_in_shape[-1]*self.centers_num))
			b_ori_feature = tf.reshape(b_input_feature,(-1,self.enc_in_shape[-1]*self.centers_num))

			attend_wx = tf.reshape(tf.nn.xw_plus_b(ori_feature, self.W_a, self.b_a),(-1,self.enc_in_shape[1],self.attention_dim))
			attend_uh_tm1 = tf.expand_dims(tf.matmul(h_tm1, self.U_a),dim=1)
			attend_e = tf.nn.tanh(tf.add(attend_wx,attend_uh_tm1))

			b_attend_wx = tf.reshape(tf.nn.xw_plus_b(b_ori_feature, self.b_W_a, self.b_b_a),(-1,self.enc_in_shape[1],self.attention_dim))
			b_attend_uh_tm1 = tf.expand_dims(tf.matmul(h_tm1, self.b_U_a),dim=1)
			b_attend_e = tf.nn.tanh(tf.add(b_attend_wx,b_attend_uh_tm1))

			attend_e = tf.matmul(tf.reshape(attend_e,(-1,self.attention_dim)),self.W) # batch_size * timestep
			attend_e = tf.nn.softmax(tf.reshape(attend_e,(-1,self.enc_in_shape[1],1)),dim=1)
			attend_e = tf.reshape(attend_e,(-1,self.enc_in_shape[1],1))


			b_attend_e = tf.matmul(tf.reshape(b_attend_e,(-1,self.attention_dim)),self.b_W) # batch_size * timestep
			b_attend_e = tf.nn.softmax(tf.reshape(b_attend_e,(-1,self.enc_in_shape[1],1)),dim=1)
			b_attend_e = tf.reshape(b_attend_e,(-1,self.enc_in_shape[1],1))

			attend_fea = tf.multiply(input_feature, attend_e)
			attend_fea = tf.reduce_sum(attend_fea,reduction_indices=1)

			b_attend_fea = tf.multiply(b_input_feature, b_attend_e)
			b_attend_fea = tf.reduce_sum(b_attend_fea,reduction_indices=1)
		
			attend_fea = tf.add(tf.matmul(attend_fea,self.A),tf.matmul(b_attend_fea,self.b_A))

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
			
			x_t = tf.gather(self.T_w2v,x_t)*tf.gather(self.T_mask,x_t)
			x_t = tf.reshape(x_t,[self.batch_size*self.beam_size,self.d_w2v])
			h = step(x_t,h_tm1)

			print('h.shape()',h.get_shape().as_list())
			drop_h = h
			predict_score_t = tf.matmul(drop_h,self.W_c) + tf.reshape(self.b_c,(1,self.voc_size))

			logprobs = tf.nn.log_softmax(predict_score_t)
			logprobs = tf.reshape(logprobs, [1, self.beam_size, self.voc_size])
			print('SamModel_test.py: decoder_onestep: test_step: logprobs:', logprobs.get_shape().as_list())

			return (h, logprobs)
		
		out_dec = test_step(x_t, h_tm1)
		return out_dec[0], out_dec[1]

	def build_model_test(self):
		self.init_parameters()
		last_output, f_encoder_output, b_encoder_output = self.encoder()
		#self.f_encoder_output = f_encoder_output
		#self.b_encoder_output = b_encoder_output
		dec_h, dec_logprobs = self.decoder_onestep()

		return (last_output, f_encoder_output, b_encoder_output), (dec_h, dec_logprobs)
		
	def build_model(self):

		self.init_parameters()
		last_output, f_encoder_output, b_encoder_output = self.encoder()
		## edited by zjc 2018/08/22
		print('SamModel_test.py: build_model: after encoder, last_output:', last_output.get_shape().as_list(),
			'f_encoder_output:', f_encoder_output.get_shape().as_list(), 'b_encoder_output:', b_encoder_output.get_shape().as_list())
		predict_score, predict_words , loss_mask= self.decoder(last_output, f_encoder_output, b_encoder_output)
		## edited by zjc 2018/08/22
		print('SamModel_test.py: build_model: after decoder, predict_score:', predict_score.get_shape().as_list(),
			'predict_words:', predict_words.get_shape().as_list(), 'loss_mask:', loss_mask.get_shape().as_list())
		print('SamModel_test.py: build_model: build beamSearchDecoder ...')
		finished_beam, logprobs_finished_beams, past_symbols = self.beamSearchDecoder(last_output, f_encoder_output, b_encoder_output)
		print('SamModel_test.py: build_model: after beamSearchDecoder, finished_beam:', finished_beam.get_shape().as_list(),
			'logprobs_finished_beams:', logprobs_finished_beams.get_shape().as_list(), 'past_symbols:', past_symbols.get_shape().as_list())
		return predict_score, loss_mask, finished_beam, logprobs_finished_beams, past_symbols


