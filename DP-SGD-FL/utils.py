import copy
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import transforms
from sampling import cifar_iid, cifar_noniid

# TODO: use chest x-ray dataset from https://github.com/NicoleStrel/Deep-Learning-with-Differential-Privacy

import pickle
import gzip
import os

def get_dataset(iid : bool, num_users : int):
	""" Returns train and test datasets and a user group which is a dict where
	the keys are the user index and the values are the corresponding data for
	each of those users.
	"""
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

	apply_transform = None

	train_dataset = ChestDataset(x_train, y_train, transform=apply_transform)
        
	#valid_dataset = ChestDataset(x_valid, y_valid, transform=apply_transform)

	test_dataset = ChestDataset(x_test, y_test, transform=apply_transform)

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

class ChestDataset(Dataset):
	def __init__(self, x, y, transform=None):
		self.x = x
		self.y = y
		self.transform = transform
	def __len__(self):
		return len(self.x)
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
        
		image = self.x[idx]
		image = transforms.ToTensor()(image)
		if self.transform:
			image = self.transform(image)
		image = transforms.Resize((64,64))(image)
		label = torch.zeros(2)
		label[self.y[idx]] = 1
		return (image, label)