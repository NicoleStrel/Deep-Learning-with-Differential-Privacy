import os
import copy
import time
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from update import LocalUpdate, test_inference
from models import CNN
from utils import get_dataset, average_weights



def federated_train(epoch):
	# Training
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
	start_time = time.time()
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
		iid=True, num_users=num_users, dataset='knee')

	# BUILD MODEL
	global_model = CNN(num_classes=5 if train_dataset.dataset == 'knee' else 2)

	# Set the model to train and send it to device.
	global_model.to(device)
	global_model.train()
	print(global_model)

	# copy weights
	global_weights = global_model.state_dict()

	# Training
	for epoch in tqdm(range(global_epochs)):
		federated_train(epoch)

	# Test inference after completion of training
	train_acc, train_loss = test_inference(cuda, global_model, train_dataset)

	valid_acc, valid_loss = test_inference(cuda, global_model, valid_dataset)

	test_acc, test_loss = test_inference(cuda, global_model, test_dataset)
	
	print(f' \n Results after {global_epochs} global rounds of training:')
	print("|---- Train Accuracy: {:.2f}%".format(100*train_acc))
	print("|---- Validation Accuracy: {:.2f}%".format(100*valid_acc))
	print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

	print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))