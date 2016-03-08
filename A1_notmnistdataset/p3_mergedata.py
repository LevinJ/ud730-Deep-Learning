# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

np.random.seed(133)
image_size = 28  # Pixel width and height.
train_size = 200000
valid_size = 10000
test_size = 18720
train_datasets = ['notMNIST_large/A.pickle', 'notMNIST_large/B.pickle', 'notMNIST_large/C.pickle', 'notMNIST_large/D.pickle', 'notMNIST_large/E.pickle', 'notMNIST_large/F.pickle', 'notMNIST_large/G.pickle', 'notMNIST_large/H.pickle', 'notMNIST_large/I.pickle', 'notMNIST_large/J.pickle']
test_datasets = ['notMNIST_small/A.pickle', 'notMNIST_small/B.pickle', 'notMNIST_small/C.pickle', 'notMNIST_small/D.pickle', 'notMNIST_small/E.pickle', 'notMNIST_small/F.pickle', 'notMNIST_small/G.pickle', 'notMNIST_small/H.pickle', 'notMNIST_small/I.pickle', 'notMNIST_small/J.pickle']


def getfilepathpickel(x):
	res = x[:-7] + '_filepath' + x[-7:]
	return res

def make_arrays(nb_rows, img_size):
	if nb_rows:
		dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
		labels = np.ndarray(nb_rows, dtype=np.int32)
	else:
		dataset, labels = None, None
	return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
	num_classes = len(pickle_files)
	filepaths_valid = np.array([])
	filepaths_train = np.array([])
	valid_dataset, valid_labels,  = make_arrays(valid_size, image_size)
	train_dataset, train_labels = make_arrays(train_size, image_size)
	vsize_per_class = valid_size // num_classes
	tsize_per_class = train_size // num_classes
		
	start_v, start_t = 0, 0
	end_v, end_t = vsize_per_class, tsize_per_class
	end_l = vsize_per_class+tsize_per_class
	for label, pickle_file in enumerate(pickle_files):			 
		try:
			with open(pickle_file, 'rb') as f:
				print(pickle_file)
				letter_set = pickle.load(f)
				f2 = open(getfilepathpickel(pickle_file), 'rb')
				filepaths = pickle.load(f2)
				
				# let's shuffle the letters to have random validation and training set
				permutated_indexes = np.random.permutation(len(letter_set))
				letter_set = letter_set[permutated_indexes]
				filepaths = filepaths[permutated_indexes]
				if valid_dataset is not None:
					valid_letter = letter_set[:vsize_per_class, :, :]
					valid_dataset[start_v:end_v, :, :] = valid_letter
# 					filepaths_valid[start_v:end_v, :, :] = filepaths[:vsize_per_class, :, :]
					filepaths_valid = np.concatenate((filepaths_valid, filepaths[:vsize_per_class]))
					valid_labels[start_v:end_v] = label
					start_v += vsize_per_class
					end_v += vsize_per_class
										
				train_letter = letter_set[vsize_per_class:end_l, :, :]
				train_dataset[start_t:end_t, :, :] = train_letter
# 				filepaths_train[start_t:end_t, :, :]  = filepaths[vsize_per_class:end_l, :, :]
				filepaths_train  = np.concatenate((filepaths_train,filepaths[vsize_per_class:end_l]))
				train_labels[start_t:end_t] = label
				start_t += tsize_per_class
				end_t += tsize_per_class
				f2.close()
		except Exception as e:
			print('Unable to process data from', pickle_file, ':', e)
			raise
		
	return valid_dataset, valid_labels, train_dataset, train_labels, filepaths_valid, filepaths_train
						


def randomize(dataset, labels, filepaths):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	shuffled_filepath = filepaths[permutation]
	return shuffled_dataset, shuffled_labels, shuffled_filepath						


def run():
	valid_dataset, valid_labels, train_dataset, train_labels, valid_filepaths, train_filepaths = merge_datasets(train_datasets, train_size, valid_size)
	_, _, test_dataset, test_labels, _, test_filepaths = merge_datasets(test_datasets, test_size)
	train_dataset, train_labels, train_filepaths = randomize(train_dataset, train_labels, train_filepaths)
	test_dataset, test_labels, test_filepaths = randomize(test_dataset, test_labels, test_filepaths)
	valid_dataset, valid_labels, valid_filepaths = randomize(valid_dataset, valid_labels, valid_filepaths)
	print('Training:', train_dataset.shape, train_labels.shape,train_filepaths.shape)
	print('Validation:', valid_dataset.shape, valid_labels.shape, valid_filepaths.shape)
	print('Testing:', test_dataset.shape, test_labels.shape, test_filepaths.shape)
	return valid_dataset, valid_labels, train_dataset, train_labels,test_dataset, test_labels, train_filepaths, test_filepaths, valid_filepaths
class MergeData:
	def merge(self):	
# 		valid_dataset, valid_labels, train_dataset, train_labels,test_dataset, test_labels 
		res = run()
		return res
	def checkoneset(self,labels):
		for i in range(10):
			print("lable ", i, "number=", (labels == i).sum())
	def checkBalance(self,res):
		valid_dataset, valid_labels, train_dataset, train_labels,test_dataset, test_labels, train_filepaths, test_filepaths, valid_filepaths = res
		self.checkoneset(valid_labels)
		return
if __name__ == "__main__":
# 	x = r'notMNIST_large/A.pickle'
# 	print (getfilepathpickel(x))
	mergeobj= MergeData()
	res = mergeobj.merge()
	mergeobj.checkBalance(res)
