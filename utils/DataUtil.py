import numpy as np
import os
import re
import h5py
import math
import json

from nltk.stem.snowball import SnowballStemmer
from collections import Counter


re_alphanumeric = re.compile('[^a-z0-9 -]+')
re_multispace = re.compile(' +')
snowball = SnowballStemmer('english') 

def preprocess_sentence(line):
	'''strip all punctuation, keep only alphanumerics
	'''
	line = re_alphanumeric.sub('', line)
	line = re_multispace.sub(' ', line)
	return line


def create_vocabulary_word2vec(file='/data/mvadD/split', capl=16, v2i={'': 0, 'UNK':1, 'BOS':2, 'EOS':3}, word_threshold=2):
	'''
	v2i = {'': 0, 'UNK':1}  # vocabulary to index
	'''
	train_list = file+'/train_split/TrainList.txt'
	train_corpus = file+'/train_split/TrainCorpus.txt'

	val_list = file+'/valid_split/ValidList.txt'
	val_corpus = file+'/valid_split/ValidCorpus.txt'

	test_list = file+'/test_split/TestList.txt'
	test_corpus = file+'/test_split/TestCorpus.txt'


	train_data = []
	val_data = []
	test_data = []


	def parse_caption(file_list,file_corpus):
		captions = []
		with open(file_list, 'r') as r_fl:
			with open(file_corpus, 'r') as r_fc:
				for vid_info in r_fl:
					vid_temp = vid_info.strip().split('/')
					vid_name = vid_temp[5]+'/'+vid_temp[7][:-4]
					# print(vid_name)
					sentence = r_fc.readline()
					# for short_sentence in sentence.strip().split('.'):
					for short_sentence in re.split(',|\.',sentence.strip()):
						if short_sentence != "":
							sentence_temp = preprocess_sentence(short_sentence.lower().strip())
							words = sentence_temp.split(' ')
							# print(len(words))
							if len(words)<capl and len(words)>=4:
								cap = {} 
								cap[vid_name] = words
								captions.append(cap)

		return captions

	
	def generate_test_data(file_list):
		captions = []

		with open(file_list, 'r') as r_fl:

			for vid_info in r_fl:
				vid_temp = vid_info.strip().split('/')
				vid_name = vid_temp[5]+'/'+vid_temp[7][:-4]
				cap = {}
				cap[vid_name] = ['']
				captions.append(cap)
		return captions

	train_data = parse_caption(train_list, train_corpus)
	val_data = parse_caption(val_list,val_corpus)

	test_data = generate_test_data(test_list)

	all_word = []
	for data in train_data:
		for k,v in data.items():
			all_word.extend(v)
	for data in val_data:
		for k,v in data.items():
			all_word.extend(v)

	vocab = Counter(all_word)
	vocab = [k for k in vocab.keys() if vocab[k] >= word_threshold]

	# create vocabulary index
	for w in vocab:
		if w not in v2i.keys():
			v2i[w] = len(v2i)


	# v2i = generate_vocab(train_data, v2i={'': 0, 'UNK':1, 'BOS':2, 'EOS':3})
	print('len v2i:',len(v2i))
	print('train %d, val %d, test %d' %(len(train_data),len(val_data),len(test_data)))
	return v2i, train_data, val_data, test_data

def create_test_ground_truth(file_list,file_corpus,output_file):

	js = {}
	with open(file_list, 'r') as r_fl:
		with open(file_corpus, 'r') as r_fc:
			for vid_info in r_fl:
				vid_temp = vid_info.strip().split('/')
				vid_name = vid_temp[5]+'/'+vid_temp[7][:-4]
				sentence = r_fc.readline()
				captions = []
				for short_sentence in re.split(',|\.',sentence.strip()):
					# print(short_sentence)
					if short_sentence != "":
						sentence_temp = preprocess_sentence(short_sentence.lower().strip())
						captions.append(sentence_temp)
				
				# captions.append(preprocess_sentence(sentence.lower().strip()))

				js[vid_name]=captions

	with open(output_file, 'w') as f:
		json.dump(js, f)

def generate_vocab(train_data, v2i={'': 0, 'UNK':1, 'BOS':2, 'EOS':3}):


	for caption_info in train_data:
		for k,v in caption_info.items():
			for w in v:
				if not v2i.has_key(w):
					v2i[w] = len(v2i)


	print('vocab size %d' %(len(v2i)))
	return v2i
	


def getBatchVideoFeature(batch_caption, hf, feature_shape):
	batch_size = len(batch_caption)
	input_video = np.zeros((batch_size,)+tuple(feature_shape),dtype='float32')

	for idx, caption in enumerate(batch_caption):
		for k,v in caption.items():
			feature = hf[k]
			input_video[idx] = np.reshape(feature,feature_shape)
	return input_video


def getBatchTrainCaption(batch_caption, v2i, capl=16):
	batch_size = len(batch_caption)

	labels = np.zeros((batch_size,capl,len(v2i)),dtype='int32')

	input_captions = np.zeros((batch_size,capl),dtype='int32')
	input_captions[:,0] = v2i['BOS']

	for idx, caption in enumerate(batch_caption):
		for vid, sen in caption.items():

			for k, w in enumerate(sen):
				
				if w in v2i.keys():
					labels[idx][k][v2i[w]] = 1
					input_captions[idx][k+1] = v2i[w]
				else:
					labels[idx][k][v2i['UNK']] = 1
					input_captions[idx][k+1] = v2i['UNK']
			# if len(sen)+1<capl:
			# 	input_captions[idx][len(sen)+1] = v2i['EOS']
			labels[idx][len(sen)][v2i['EOS']] = 1
	# print(batch_caption)
	# print(input_captions)
	# print(np.sum(labels,-1))
	return input_captions, labels

def getNewBatchTrainCaption(batch_caption, v2i, capl=16):
	batch_size = len(batch_caption)

	

	input_captions = np.zeros((batch_size,capl),dtype='int32')
	input_captions[:,0] = v2i['BOS']

	for idx, caption in enumerate(batch_caption):
		for vid, sen in caption.items():

			for k, w in enumerate(sen):
				
				if w in v2i.keys():
					input_captions[idx][k+1] = v2i[w]
				else:
					input_captions[idx][k+1] = v2i['UNK']

	labels = np.zeros((batch_size,capl,len(v2i)),dtype='int32')
	for i, sentence in enumerate(input_captions):
		for j, word in enumerate(sentence):
			if j>=1:
				if word != 0:
					labels[i,j-1,word]=1
				elif word == 0:
					labels[i,j-1,v2i['EOS']]=1
					# break
				# labels[i,j-1,word]=1

	# print(batch_caption)
	# print(input_captions)
	# print(np.sum(labels,-1))
	return input_captions, labels
def getBatchTestCaption(batch_caption, v2i, capl=16):
	batch_size = len(batch_caption)
	labels = np.zeros((batch_size,capl,len(v2i)),dtype='int32')
	input_captions = np.zeros((batch_size,capl),dtype='int32')
	input_captions[:,0] = v2i['BOS']


	return input_captions, labels


def convertCaptionI2V(batch_caption, generated_captions,i2v):
	captions = []
	for idx, sen in enumerate(generated_captions):
		caption = ''
		for word in sen:
			if i2v[word]=='EOS':
				break
			caption+=i2v[word]+' '
		captions.append(caption)
	return captions
if __name__=='__main__':
	# create_vocabulary_word2vec()
	file='/home/xyj/usr/local/data/mvad/M-VAD/split'
	test_list = file+'/test_split/TestList.txt'
	test_corpus = file+'/test_split/TestCorpus.txt'
	output_file = '/home/xyj/usr/local/data/mvad/mvad_reference.json'
	create_test_ground_truth(test_list,test_corpus,output_file)