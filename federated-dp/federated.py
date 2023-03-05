import os
import copy
import time
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from update import LocalUpdate, test_inference
from models import CNNCifar
from utils import get_dataset, average_weights


if __name__ == '__main__':
	start_time = time.time()
	cuda = False
	num_users = 100
	frac = 0.1
	global_epochs = 10
	local_epochs = 10

	# define paths
	path_project = os.path.abspath('..')
	logger = SummaryWriter('../logs')

	device = 'cuda' if cuda else 'cpu'

	# load dataset and user groups
	train_dataset, test_dataset, user_groups = get_dataset(iid=True, 
														   num_users=num_users)

	# BUILD MODEL
	global_model = CNNCifar(num_classes=10)

	# Set the model to train and send it to device.
	global_model.to(device)
	global_model.train()
	print(global_model)

	# copy weights
	global_weights = global_model.state_dict()

	# Training
	train_loss, train_accuracy = [], []
	val_acc_list, net_list = [], []
	cv_loss, cv_acc = [], []
	print_every = 2
	val_loss_pre, counter = 0, 0

	for epoch in tqdm(range(global_epochs)):
		local_weights, local_losses = [], []
		print(f'\n | Global Training Round : {epoch+1} |\n')

		global_model.train()
		m = max(int(frac * num_users), 1)
		idxs_users = np.random.choice(range(num_users), m, replace=False)

		for idx in idxs_users:
			local_model = LocalUpdate(cuda=cuda, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger,
									  batch_size=10, lr=0.01, 
									  epochs=local_epochs)
			w, loss = local_model.update_weights(
				model=copy.deepcopy(global_model), global_round=epoch)
			local_weights.append(copy.deepcopy(w))
			local_losses.append(copy.deepcopy(loss))

		# update global weights
		global_weights = average_weights(local_weights)

		# update global weights
		global_model.load_state_dict(global_weights)

		loss_avg = sum(local_losses) / len(local_losses)
		train_loss.append(loss_avg)

		# Calculate avg training accuracy over all users at every epoch
		list_acc, list_loss = [], []
		global_model.eval()
		for c in range(num_users):
			local_model = LocalUpdate(cuda=cuda, dataset=train_dataset,
									  idxs=user_groups[idx], logger=logger)
			acc, loss = local_model.inference(model=global_model)
			list_acc.append(acc)
			list_loss.append(loss)
		train_accuracy.append(sum(list_acc)/len(list_acc))

		# print global training loss after every 'i' rounds
		if (epoch+1) % print_every == 0:
			print(f' \nAvg Training Stats after {epoch+1} global rounds:')
			print(f'Training Loss : {np.mean(np.array(train_loss))}')
			print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

	# Test inference after completion of training
	test_acc, test_loss = test_inference(cuda, global_model, test_dataset)

	print(f' \n Results after {global_epochs} global rounds of training:')
	print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
	print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

	print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

	# PLOTTING
	import matplotlib
	import matplotlib.pyplot as plt
	matplotlib.use('Agg')

	# Plot Loss curve
	plt.figure()
	plt.title('Training Loss vs Communication rounds')
	plt.plot(range(len(train_loss)), train_loss, color='r')
	plt.ylabel('Training loss')
	plt.xlabel('Communication Rounds')
	plt.show()
	
	# Plot Average Accuracy vs Communication rounds
	plt.figure()
	plt.title('Average Accuracy vs Communication rounds')
	plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
	plt.ylabel('Average Accuracy')
	plt.xlabel('Communication Rounds')
	plt.show()