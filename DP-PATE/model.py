import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Class used to initialize CNN model of student/teacher"""

    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 32, 3)
        self.fc1 = nn.Linear(32 * 14 * 14, 64 * num_classes)
        self.fc2 = nn.Linear(64 * num_classes, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
