import copy
import torch
from torchvision import datasets, transforms
from sampling import cifar_iid, cifar_noniid

# TODO: use chest x-ray dataset from https://github.com/NicoleStrel/Deep-Learning-with-Differential-Privacy

import pickle
import gzip
import os

path = 'chest-data' #os.path.join(os.path.dirname(os.getcwd()), 'chest-data')

with gzip.open(os.path.join(path, 'chest_x_train.gz'), 'rb') as i:
    x_train = pickle.load(i)
with gzip.open(os.path.join(path, 'chest_x_val.gz'), 'rb') as i:
    x_valid = pickle.load(i)    
with gzip.open(os.path.join(path, 'chest_x_test.gz'), 'rb') as i:
    x_test = pickle.load(i)  
with gzip.open(os.path.join(path, 'chest_y_train.gz'), 'rb') as i:
    y_train = pickle.load(i)  
with gzip.open(os.path.join(path, 'chest_y_val.gz'), 'rb') as i:
    y_valid = pickle.load(i) 
with gzip.open(os.path.join(path, 'chest_y_test.gz'), 'rb') as i:
    y_test = pickle.load(i)

def get_dataset(iid : bool, num_users : int):
	""" Returns train and test datasets and a user group which is a dict where
	the keys are the user index and the values are the corresponding data for
	each of those users.
	"""
	apply_transform = transforms.Compose(
		[transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	train_dataset = datasets.CIFAR10("./cifar10", train=True, download=True, transform=apply_transform)

	test_dataset = datasets.CIFAR10("./cifar10", train=False, download=True, transform=apply_transform)

	# sample training data amongst users
	if iid:
		user_groups = cifar_iid(train_dataset, num_users)
	else:
		user_groups = cifar_noniid(train_dataset, num_users)

	return train_dataset, test_dataset, user_groups


def average_weights(w):
	"""
	Returns the average of the weights.
	"""
	w_avg = copy.deepcopy(w[0])
	for key in w_avg.keys():
		for i in range(1, len(w)):
			w_avg[key] += w[i][key]
		w_avg[key] = torch.div(w_avg[key], len(w))
	return w_avg