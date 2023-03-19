from torch import nn
import torch.nn.functional as F
	
class CNN(nn.Module):
	def __init__(self, num_classes):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 32, 3)
		self.fc1 = nn.Linear(6272, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, num_classes)
	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 6272)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return self.fc3(x) 