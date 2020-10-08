import numpy as np
import os
import h5py
import math

from utils import SeqVladDataUtil_object as SeqVladDataUtil
#from model import SamModel 
from model import SamModel_ObjectV_att_NoShare_test as SamModel

import tensorflow as tf
import cPickle as pickle
import time
import json
import argparse


parser = argparse.ArgumentParser(description='seqvlad, youtube, video captioning, reduction app')

parser.add_argument('--soft', action='store_true',
						help='soft method to train')
parser.add_argument('--bidirectional', action='store_true',
                                                help='new bidirectional')
parser.add_argument('--step', action='store_true',
						help='step training')
parser.add_argument('--gpu_id', type=str, default="0",
						help='specify gpu id')
parser.add_argument('--input_feature', type=str, default="",
						help='path to input feature')
parser.add_argument('--input_feature_dim', type=int, default=2048,
						help='dimension of input feature')

parser.add_argument('--model_folder', type=str, default="youtube",
						help='name of folder to save models')
parser.add_argument('--cap_len_max', type=int, default=16,
						help='the maximum of the caption length')
parser.add_argument('--cap_len_min', type=int, default=1,
						help='the minimum of the caption length')
parser.add_argument('--lr', type=float, default=0.0001,
						help='learning reate')
parser.add_argument('--epoch', type=int, default=20,
						help='total runing epoch')
parser.add_argument('--d_w2v', type=int, default=512,
						help='the dimension of word 2 vector')
parser.add_argument('--output_dim', type=int, default=512,
						help='the hidden size')
parser.add_argument('--centers_num', type=int, default=16,
						help='the number of centers')
parser.add_argument('--reduction_dim', type=int, default=512,
						help='the reduction dim of input feature, e.g., 1024->512')
parser.add_argument('--bottleneck', type=int, default=256,
						help='the bottleneck size')
parser.add_argument('--pretrained_model', type=str, default=None,
						help='the pretrained model')
args = parser.parse_args()
w1 = 0.8
w2 = 0.2

def evaluate_mode_by_shell(res_path,js):
	with open(res_path, 'w') as f:
		json.dump(js, f)

	command ='caption_eval/call_python_caption_eval.sh '+ '\"' + os.path.abspath(res_path) + '\"'
	os.system(command)

def main(hf,obj_hfs,f_type,
		reduction_dim=512, centers_num = 32, kernel_size=1, capl_l=16, capl_s=1, d_w2v=512, output_dim=512,
		batch_size=64, total_epoch=args.epoch,
		file=None, obj_file=None, obj_file_rev=None, saveprefix='youtube',beam_size=5, obj_num=5):

	print('main: batch_size = %d' % batch_size)
	# Create vocabulary
	timesteps_v = feature_shape[0]
	fetaure_dim = feature_shape[1]
	capl = capl_l
	v2i, train_data, val_data, test_data = SeqVladDataUtil.create_vocabulary_word2vec_minmax(file, capl_max=capl_l, capl_min=capl_s, v2i={'': 0, 'UNK':1,'BOS':2, 'EOS':3})
	i2v = {i:v for v,i in v2i.items()}

	voc_size = len(v2i)

	#configure && runtime environment
	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 1.0
	config.log_device_placement=False

	model_path1 = 'saved_model/msvd/youtube_sample40_objSim_NMS0.3_WV_att_NoShare_resnet200_res5c_relu_capl15s3/soft_capl15s3_dw2v512512_c64_redu512_lr0.0001_B16/model/E3_L2.218330876083131.ckpt'
	model_path2 = 'saved_model/msvd/youtube_sample40_objSim_NMS0.3_reverse_WV_att_NoShare_resnet200_res5c_relu_capl15s3/soft_capl15s3_dw2v512512_c64_redu512_lr0.0001_B16/model/E8_L1.6779830772685718.ckpt'

	g1 = tf.Graph()
	sess1 = tf.Session(config=config, graph=g1)
	with sess1.as_default():
		with g1.as_default():
			input_video1 = tf.placeholder(tf.float32, shape=(None,)+feature_shape,name='input_video')
			input_object1 = tf.placeholder(tf.float32, shape=(None, obj_num)+feature_shape,name='input_object')
			input_captions1 = tf.placeholder(tf.int32, shape=(None,capl), name='input_captions')
			dec_x_t_1 = tf.placeholder(tf.int32, shape=(None,None), name='dec_x_t')
			dec_h_tm1_1 = tf.placeholder(tf.float32, shape=(None, output_dim), name='dec_h_tm1')
			dec_v_in_feature1 = tf.placeholder(tf.float32, shape=(None, timesteps_v, reduction_dim*centers_num), name='dec_v_in_feature')
			dec_o_in_feature1 = tf.placeholder(tf.float32, shape=(obj_num, None, timesteps_v, reduction_dim*centers_num), name='dec_o_in_feature')
			captionModel1 = SamModel.SoftModel(input_video1, input_object1, input_captions1, dec_x_t_1, dec_h_tm1_1, dec_v_in_feature1, dec_o_in_feature1,
									voc_size, d_w2v, output_dim,
									reduction_dim=reduction_dim,
									centers_num=centers_num, 
									done_token=v2i['EOS'], max_len = capl, beamsearch_batchsize = 1, beam_size=5)
			out_enc1, out_dec1 = captionModel1.build_model_test()
			## out_enc: (last_output1, f_vlad1, b_vlad1)
			## out_dec: (dec_h1, dec_logprobs1)
			init = tf.global_variables_initializer()
			sess1.run(init)
			saver1 = tf.train.Saver(sharded=True,max_to_keep=total_epoch)
			saver1.restore(sess1, model_path1)


	g2 = tf.Graph()
	sess2 = tf.Session(config=config, graph=g2)
	with sess2.as_default():
		with g2.as_default():
			input_video2 = tf.placeholder(tf.float32, shape=(None,)+feature_shape,name='input_video')
			input_object2 = tf.placeholder(tf.float32, shape=(None, obj_num)+feature_shape,name='input_object')
			input_captions2 = tf.placeholder(tf.int32, shape=(None,capl), name='input_captions')	
			dec_x_t_2 = tf.placeholder(tf.int32, shape=(None,None), name='dec_x_t')
			dec_h_tm1_2 = tf.placeholder(tf.float32, shape=(None, output_dim), name='dec_h_tm1')
			dec_v_in_feature2 = tf.placeholder(tf.float32, shape=(None, timesteps_v, reduction_dim*centers_num), name='dec_v_in_feature')
			dec_o_in_feature2 = tf.placeholder(tf.float32, shape=(obj_num, None, timesteps_v, reduction_dim*centers_num), name='dec_o_in_feature')
			captionModel2 = SamModel.SoftModel(input_video2, input_object2, input_captions2, dec_x_t_2, dec_h_tm1_2, dec_v_in_feature2, dec_o_in_feature2,
									voc_size, d_w2v, output_dim,
									reduction_dim=reduction_dim,
									centers_num=centers_num, 
									done_token=v2i['EOS'], max_len = capl, beamsearch_batchsize = 1, beam_size=5)
			out_enc2, out_dec2 = captionModel2.build_model_test()
			init = tf.global_variables_initializer()
			sess2.run(init)
			saver2 = tf.train.Saver(sharded=True,max_to_keep=total_epoch)
			saver2.restore(sess2, model_path2)

	'''
	sess3 = tf.Session(config=config)
	with sess3.as_default():
		init = tf.global_variables_initializer()
		sess3.run(init)
	'''
	def perform_fusion(w1, w2):
		batch_size = 1
		data = test_data

		caption_output = []
		total_data = len(data)
		num_batch = int(round(total_data*1.0/batch_size))

		for batch_idx in xrange(num_batch):
			batch_caption = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
			data_v, data_obj = SeqVladDataUtil.getBatchVideoObjectFeature2(batch_caption,hf,obj_hfs,obj_num,feature_shape,obj_file)
			_, data_obj_rev = SeqVladDataUtil.getBatchVideoObjectFeature2(batch_caption,hf,obj_hfs,obj_num,feature_shape,obj_file_rev)
			data_c, data_y = SeqVladDataUtil.getBatchTestCaptionWithSparseLabel(batch_caption, v2i, capl=capl)
			
			#out_enc_g1 = sess1.run(out_enc1, feed_dict={input_video1:data_v, input_captions1:data_c})
			out_enc_g1 = sess1.run(out_enc1, feed_dict={input_video1:data_v, input_object1:data_obj})
			(last_output1, v_enc_out1, o_enc_out1) = out_enc_g1
			assert(last_output1.shape[0]==1)

			#out_enc_g2 = sess2.run(out_enc2, feed_dict={input_video2:data_v, input_captions2:data_c})
			out_enc_g2 = sess2.run(out_enc2, feed_dict={input_video2:data_v, input_object2:data_obj_rev})
			(last_output2, v_enc_out2, o_enc_out2) = out_enc_g2
			assert(last_output2.shape[0]==1)
			assert(last_output2.shape[1]==output_dim)

			x_0 = data_c[:,0]
			x_0 = np.expand_dims(x_0,axis=-1)
			x_0 = np.tile(x_0, [1, beam_size])
			h_0_1 = np.expand_dims(last_output1, axis=1)
			h_0_1 = np.reshape(np.tile(h_0_1, [1, beam_size, 1]), (beam_size, output_dim))
			
			h_0_2 = np.expand_dims(last_output2, axis=1)
			h_0_2 = np.reshape(np.tile(h_0_2, [1, beam_size, 1]), (beam_size, output_dim))
			
			x_t = x_0
			h_tm1_1 = h_0_1
			h_tm1_2 = h_0_2
			
			finished_beams = np.zeros((batch_size, capl), dtype=np.int32) # shape [1, capl]
			logprobs_finished_beams = np.ones((batch_size,), dtype=np.float32) * float('inf') # shape [1,]
			
			for time in range(capl):
				###
				out_dec_g1 = sess1.run(out_dec1, feed_dict={dec_x_t_1:x_t, dec_h_tm1_1:h_tm1_1,
																										dec_v_in_feature1:v_enc_out1, dec_o_in_feature1:o_enc_out1})
				(h_t_1, logprobs1) = out_dec_g1
				
				out_dec_g2 = sess2.run(out_dec2, feed_dict={dec_x_t_2:x_t, dec_h_tm1_2:h_tm1_2,
																										dec_v_in_feature2:v_enc_out2, dec_o_in_feature2:o_enc_out2})
				(h_t_2, logprobs2) = out_dec_g2
				
				###
				logprobs = w1*logprobs1 + w2*logprobs2 # shape [beam_size, voc_size]

				
				if time == 0:
					logprobs_batched = np.reshape(logprobs, [-1, beam_size, voc_size])
					t_logprobs = logprobs_batched[:,0,:] # shape [1, voc_size]
					desc_ind = np.argsort(-t_logprobs, axis=1)
					topk_indices = desc_ind[0,:beam_size] # shape [beam_size, ]
					past_logprobs = t_logprobs[:,topk_indices] # shape [1, beam_size]
					topk_indices = np.reshape(topk_indices, [1, beam_size])
					#past_logprobs, topk_indices = tf.nn.top_k(logprobs_batched[:,0,:], beam_size)
				else:
					logprobs = np.reshape(logprobs, [-1, beam_size, voc_size])
					logprobs = logprobs+np.expand_dims(past_logprobs, axis=2) # shape [1, beam_size, voc_size]
					t_logprobs = np.reshape(logprobs, [1, beam_size*voc_size])
					desc_ind = np.argsort(-t_logprobs, axis=1)
					topk_indices = desc_ind[0,:beam_size] # shape [beam_size, ]
					past_logprobs = t_logprobs[:,topk_indices] # shape [1, beam_size]
					topk_indices = np.reshape(topk_indices, [1, beam_size])
					#past_logprobs, topk_indices = tf.nn.top_k(
					#	tf.reshape(logprobs, [1, beam_size * voc_size]),
					#	beam_size, 
					#	sorted=False
					#)
				symbols = topk_indices % voc_size
				symbols = np.reshape(symbols, [1, beam_size]) # shape [1, beam_size]
				parent_refs = topk_indices // voc_size
				parent_refs = np.reshape(parent_refs, [-1])
				#h_1 = tf.gather(h_t_1,  tf.reshape(parent_refs,[-1]))
				#h_2 = tf.gather(h_t_2,  tf.reshape(parent_refs,[-1]))
				h_1 = h_t_1[parent_refs]
				h_2 = h_t_2[parent_refs]
				done_token=v2i['EOS']
				
				if time==0:
					past_symbols = np.concatenate([np.expand_dims(symbols, axis=2), np.zeros((batch_size, beam_size, capl-1), dtype=np.int32)],axis=-1)
					# shape [1, beam_size, capl]
				else:
					past_symbols_batch_major = np.reshape(past_symbols[:,:,0:time], [-1, time]) # shape [beam_size, time]
					#beam_past_symbols = tf.gather(past_symbols_batch_major,  parent_refs)
					beam_past_symbols = np.reshape(past_symbols_batch_major[parent_refs], [-1, beam_size, time]) # shape [1, beam_size, time]			
					past_symbols = np.concatenate([beam_past_symbols, np.expand_dims(symbols, axis=2), np.zeros((1, beam_size, capl-time-1), dtype=np.int32)],axis=2)
					past_symbols = np.reshape(past_symbols, [1,beam_size,capl])
					
					cond1 = np.equal(symbols, np.ones(symbols.shape, dtype=np.int32)*done_token) # condition on done sentence
					# cond1: shape [1, beam_size]
				
					#for_finished_logprobs = tf.where(cond1,past_logprobs,tf.ones_like(past_logprobs,tf.float32)* -1e5)
					for_finished_logprobs = np.where(cond1, past_logprobs, np.ones(past_logprobs.shape, np.float32)* -1e5) # shape [1, beam_size]
				
					#done_indice_max = tf.cast(tf.argmax(for_finished_logprobs,axis=-1),tf.int32) # shape [1,]
					#logprobs_done_max = tf.reduce_max(for_finished_logprobs,reduction_indices=-1) # shape [1,]
					#done_past_symbols = tf.gather(tf.reshape(past_symbols,[beam_size,capl]),done_indice_max) # shape [1, capl]
					#logprobs_done_max = tf.div(-logprobs_done_max,tf.cast(time,tf.float32))
					#cond2 = tf.greater(logprobs_finished_beams,logprobs_done_max) # shape [1,]

					#cond3 = tf.equal(done_past_symbols[:,time],done_token)
					#cond4 = tf.equal(time,capl-1)
					#finished_beams = tf.where(tf.logical_and(cond2,tf.logical_or(cond3,cond4)),
				  #															done_past_symbols,
				  #															finished_beams)
					#logprobs_finished_beams = tf.where(tf.logical_and(cond2,tf.logical_or(cond3,cond4)),
					#							logprobs_done_max, 
					#							logprobs_finished_beams)

					done_indice_max = int(np.argmax(np.reshape(for_finished_logprobs, [beam_size]))) # int32 
					logprobs_done_max = for_finished_logprobs[:, done_indice_max] # shape [1,]
					done_past_symbols = past_symbols[:,done_indice_max,:] # shape [1, capl]
					logprobs_done_max = -logprobs_done_max / float(time) # shape [1, ]
					cond2 = np.greater(logprobs_finished_beams, logprobs_done_max) # shape [1,]
					cond3 = np.equal(done_past_symbols[:,time], done_token) # shape [1, ]
					cond4 = np.equal(time,capl-1) # bool
					finished_beams = np.where(cond2 and (cond3 or cond4),
				  															done_past_symbols,
				  															finished_beams)
					logprobs_finished_beams = np.where(cond2 and (cond3 or cond4),
												logprobs_done_max, 
												logprobs_finished_beams)

				x_t = symbols
				h_tm1_1 = h_1
				h_tm1_2 = h_2


			#fb = sess3.run(finished_beams)
			fb = finished_beams
			generated_captions = SeqVladDataUtil.convertCaptionI2V(batch_caption, fb, i2v)

			for idx, sen in enumerate(generated_captions):
				print('%s : %s' %(batch_caption[idx].keys()[0],sen))
				caption_output.append({'image_id':batch_caption[idx].keys()[0],'caption':sen})
		
		js = {}
		js['val_predictions'] = caption_output

		res_path = 'fusion/%s_w1(%.2f)_w2(%.2f).json' % (saveprefix, w1, w2)
		evaluate_mode_by_shell(res_path,js)
	w1 = 1.0
	w2 = 0.6
	perform_fusion(w1, w2)
	
	'''
	for w1_i in range(5):
		w1 = (5-w1_i) * 0.2
		#w1 = 0
		for w2_i in range(5):
			w2 = (5-w2_i) * 0.2
			print('now w1 = %f, w2 = %f' % (w1,w2))
			perform_fusion(w1, w2)
	'''


if __name__ == '__main__':
	args.centers_num=64
	args.lr=0.0001
	args.epoch=20
	args.d_w2v=512
	args.output_dim=512
	args.reduction_dim=512
	args.gpu_id='9'
	args.cap_len_max=15
	args.cap_len_min=3
	args.input_feature_dim=2048
	args.input_feature = 'features/msvd/msvd_all_sample40_frame_resnet200_res5c_relu.h5'
	print(args)
	os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
	d_w2v = args.d_w2v # the dimension of word 2 vector, set to 512
	output_dim = args.output_dim # hidden size, set to 512
	reduction_dim=args.reduction_dim # the reduction dim of input feature, e.g., 1024->512, set to 512
	centers_num = args.centers_num # the number of centers, set to 64
	bottleneck = args.bottleneck # the bottleneck size, set to 256

	#capl = 16
	capl_l = args.cap_len_max+1
	capl_s = args.cap_len_min
	

	feature_path = args.input_feature
	save_model_dir = args.model_folder

	f_type = 'capl%ds%d_'%(args.cap_len_max, args.cap_len_min) + 'dw2v'+str(d_w2v)+str(output_dim)+'_c'+str(centers_num)+'_redu'+str(reduction_dim)
	
	#video_feature_dims = 2048
	#video_feature_dims = 1024
	video_feature_dims = args.input_feature_dim
	timesteps_v = 40 # sequences length for video
	height = 7
	width = 7
	feature_shape = (timesteps_v,video_feature_dims,height,width)
	
	print('feature_path: %s, video_feature_dims: %d, timesteps_v = %d' % (feature_path, video_feature_dims, timesteps_v))

	#feature_path = '/home/zhangjunchao/workspace/Caption/MSVD/features/msvd_all_sample'+str(timesteps_v)+'_frame_googlenet_bn_in5b'+'.h5'
	#feature_path = '/home/zhangjunchao/workspace/Caption/MSVD/features/msvd_all_sample40_frame_resnet200_res5c_relu.h5'
	#feature_path = '/home/zhangjunchao/workspace/Caption/MSVD/features/msvd_all_sample40_frame_msdn_obj_top49_aaai19.h5'
	
	#save_model_dir = 'youtube_sample40_resnet200'
	if args.step:
		f_type = 'step_'+ f_type
	
	hf = h5py.File(feature_path,'r')
	
	obj_hfs = []
	obj_num_0 = 5
	for obj_i in range(obj_num_0):
		obj_h5path = 'features/msvd/features_msvd_sample40_nms0.3_box%d.h5' % (obj_i+1)
		obj_hfs.append(h5py.File(obj_h5path,'r'))
	obj_num = 5

	if args.soft:
		f_type = 'soft_'+f_type
	else:
		f_type = 'hard_'+f_type
	obj_file = 'features/msvd/SimilarityGraph/listAll_nms0.3/'
	obj_file_rev = 'features/msvd/SimilarityGraph/listAll_nms0.3_reverse/'
	
	batch_size = 32
	saveprefix = 'objSim_NMS0.3_WV_att_NoShare_capl15s3_FwRevE3E8_right'
	print('cap_len: [%d, %d)\saveprefix: %s\nf_type: %s' % (capl_s, capl_l, saveprefix, f_type))
	main(hf,obj_hfs,f_type, 
		reduction_dim=reduction_dim,centers_num=centers_num, capl_l=capl_l, capl_s=capl_s, 
		d_w2v=d_w2v, output_dim=output_dim, batch_size=batch_size, #bottleneck=bottleneck,
		file='./data/msvd', obj_file=obj_file, obj_file_rev=obj_file_rev, saveprefix=saveprefix, beam_size=5, obj_num=obj_num)




	

	
	
	
	


	
