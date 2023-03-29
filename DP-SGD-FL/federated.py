import os
import copy
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from update import LocalUpdate, test_inference
from models import CNN
from utils import get_dataset, average_weights
import torch.optim as optim
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
from metrics_calc_helper_functions import *

def non_federated_train(model, train_dataset):
	model.train()
	optimizer = optim.SGD(model.parameters(), lr=0.01)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100 if train_dataset.dataset == 'chest' else 300, shuffle=True)
	loss_fn = torch.nn.CrossEntropyLoss()
	
	for _ in tqdm(range(global_epochs)):
		for (data, target) in train_loader:
			optimizer.zero_grad()
			output = model(data)
			loss = loss_fn(output, target)
			loss.backward()
			optimizer.step()

def federated_train(global_model):
	global_model.train()
	# Training
	for epoch in tqdm(range(global_epochs)):
		local_weights, local_losses = [], []
		print(f'\n | Global Training Round : {epoch+1} |\n')

		global_model.train()
		m = max(int(frac * num_users), 1)
		idxs_users = np.random.choice(range(num_users), m, replace=False)

		for idx in idxs_users:
			local_model = LocalUpdate(cuda=cuda, dataset=train_dataset, 
					epochs=local_epochs, idxs=user_groups[idx], logger=logger,
					batch_size=int(100/num_users) if train_dataset.dataset == 'chest' else int(300/num_users))
			w, loss = local_model.update_weights(
				model=copy.deepcopy(global_model), global_round=epoch)
			local_weights.append(copy.deepcopy(w))
			local_losses.append(copy.deepcopy(loss))

		# update global weights
		global_weights = average_weights(local_weights)

		# update global weights
		global_model.load_state_dict(global_weights)

if __name__ == '__main__':
	cuda = False
	num_users = 10
	frac = 0.1
	global_epochs = 10
	local_epochs = 10

	# define paths
	path_project = os.path.abspath('..')
	logger = SummaryWriter('../logs')

	device = 'cuda' if cuda else 'cpu'

	# load dataset and user groups
	train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(
		dataset='knee', num_users=num_users, iid=True)

	# BUILD MODEL
	global_model = CNN(num_classes=5 if train_dataset.dataset == 'knee' else 2)
	model = CNN(num_classes=5 if train_dataset.dataset == 'knee' else 2)

	# Set the model to train and send it to device.
	global_model.to(device)
	model.to(device)

	# copy weights
	global_weights = global_model.state_dict()

	# Training
	fed_runtime, fed_peak_mem, fed_result = get_memory_usage_and_runtime(federated_train, (global_model,))
	epsilon = get_epsilon_momentents_gaussian_dp(len(train_dataset), 4, 10, batch_size=100 if train_dataset.dataset == 'chest' else 300)
	runtime, peak_mem, result = get_memory_usage_and_runtime(non_federated_train, (model, train_dataset))
	# Test inference after completion of training
	train_acc, train_loss = test_inference(cuda, global_model, train_dataset)

	valid_acc, valid_loss = test_inference(cuda, global_model, valid_dataset)

	fed_test_acc, test_loss = test_inference(cuda, global_model, test_dataset)
	
	print(f' \n Results (federated) after {global_epochs} global rounds of training:')
	print("|---- Train Accuracy: {:.2f}%".format(100*train_acc))
	print("|---- Validation Accuracy: {:.2f}%".format(100*valid_acc))
	print("|---- Test Accuracy: {:.2f}%".format(100*fed_test_acc))

	# Test inference after completion of training
	train_acc, train_loss = test_inference(cuda, model, train_dataset)

	valid_acc, valid_loss = test_inference(cuda, model, valid_dataset)

	test_acc, test_loss = test_inference(cuda, model, test_dataset)
	
	print(f' \n Results (normal) after {global_epochs} rounds of training:')
	print("|---- Train Accuracy: {:.2f}%".format(100*train_acc))
	print("|---- Validation Accuracy: {:.2f}%".format(100*valid_acc))
	print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

	dump_metrics_to_json('federated_knee.txt', fed_runtime, fed_peak_mem, fed_test_acc*100, epsilon, True)
	dump_metrics_to_json('non_federated_knee.txt', runtime, peak_mem, test_acc*100)