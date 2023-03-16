import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
	"""An abstract Dataset class wrapped around Pytorch Dataset class.
	"""

	def __init__(self, dataset, idxs):
		self.dataset = dataset
		self.idxs = [int(i) for i in idxs]

	def __len__(self):
		return len(self.idxs)

	def __getitem__(self, item):
		image, label = self.dataset[self.idxs[item]]
		return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
	def __init__(self, cuda, dataset, idxs, logger, batch_size=32, lr=0.005, 
				 epochs=20):
		self.logger = logger
		self.batch_size = batch_size
		self.lr = lr
		self.epochs = epochs
		self.trainloader, self.validloader, self.testloader = self.train_val_test(
			dataset, list(idxs))
		self.device = 'cuda' if cuda else 'cpu'
		self.criterion = nn.CrossEntropyLoss().to(self.device)
		self.C = 35
		self.sigma = 4

	def train_val_test(self, dataset, idxs):
		"""
		Returns train, validation and test dataloaders for a given dataset
		and user indexes.
		"""
		# split indexes for train, validation, and test (80, 10, 10)
		idxs_train = idxs[:int(0.8*len(idxs))]
		idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
		idxs_test = idxs[int(0.9*len(idxs)):]

		trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
								 batch_size=self.batch_size, shuffle=True)
		validloader = DataLoader(DatasetSplit(dataset, idxs_val),
								 batch_size=int(len(idxs_val)/10), shuffle=False)
		testloader = DataLoader(DatasetSplit(dataset, idxs_test),
								batch_size=int(len(idxs_test)/10), shuffle=False)
		return trainloader, validloader, testloader

	def update_weights(self, model, global_round):
		# Set mode to train model
		model.train()
		epoch_loss = []
		g = []

		for iter in range(self.epochs):
			batch_loss = []

			for batch_idx, batch in enumerate(self.trainloader):
				images, labels = batch

				for image, label in zip(images, labels):
					image, label = image.to(self.device), label.to(self.device)
					model.zero_grad()
					log_probs = model(image)
					loss = self.criterion(log_probs, label.unsqueeze(0))
					loss.backward()

					for param in model.parameters():
						grad = param.grad.detach().clone()
						# clip gradients
						norm = torch.norm(grad)
						g.append(norm.item())
						grad_clipped = grad / max(1, norm / self.C)
						# add Gaussian noise
						grad_clipped += torch.normal(0, self.sigma, size=grad_clipped.shape)
						# update weights
						param.data.add_(-self.lr, grad_clipped)

				if batch_idx % 10 == 0:
					print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
						global_round, iter, batch_idx * len(images),
						len(self.trainloader.dataset),
						100. * batch_idx / len(self.trainloader), loss.item()))
				self.logger.add_scalar('loss', loss.item())
				batch_loss.append(loss.item())
			epoch_loss.append(sum(batch_loss)/len(batch_loss))
		print("***********************************************",sum(g)/len(g))
		return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

	def inference(self, model):
		""" Returns the inference accuracy and loss.
		"""

		model.eval()
		loss, total, correct = 0.0, 0.0, 0.0

		for batch_idx, (images, labels) in enumerate(self.testloader):
			images, labels = images.to(self.device), labels.to(self.device)

			# Inference
			outputs = model(images)
			batch_loss = self.criterion(outputs, labels)
			loss += batch_loss.item()

			# Prediction
			_, pred_labels = torch.max(outputs, 1)
			pred_labels = pred_labels.view(-1)
			correct += torch.sum(torch.eq(pred_labels, labels)).item()
			total += len(labels)

		accuracy = correct/total
		return accuracy, loss


def test_inference(cuda, model, test_dataset):
	""" Returns the test accuracy and loss.
	"""

	model.eval()
	loss, total, correct = 0.0, 0.0, 0.0

	device = 'cuda' if cuda else 'cpu'
	criterion = nn.NLLLoss().to(device)
	testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

	for batch_idx, (images, labels) in enumerate(testloader):
		images, labels = images.to(device), labels.to(device)

		# Inference
		outputs = model(images)
		batch_loss = criterion(outputs, labels)
		loss += batch_loss.item()

		# Prediction
		_, pred_labels = torch.max(outputs, 1)
		pred_labels = pred_labels.view(-1)
		correct += torch.sum(torch.eq(pred_labels, labels)).item()
		total += len(labels)

	accuracy = correct/total
	return accuracy, loss