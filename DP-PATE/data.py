import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

import copy
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import transforms

import pickle
import gzip
import os


def load_data(train, batch_size, dataset: str):
    """Helper function used to load the train/test data.
       Args:
           train[boolean]: Indicates whether its train/test data.
           batch_size[int]: Batch size
    """
    # loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         "../data",
    #         train=train,
    #         download=True,
    #         transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    #         ),
    #     ),
    #     batch_size=batch_size,
    #     shuffle=True,
    # )
    #
    # return loader

    if dataset == 'chest':
        path = 'chest-data'  # os.path.join(os.path.dirname(os.getcwd()), 'chest-data')
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
    else:
        path = 'knee-data'  # os.path.join(os.path.dirname(os.getcwd()), 'knee-data')
        with gzip.open(os.path.join(path, 'knee_x_train.gz'), 'rb') as i:
            x_train = pickle.load(i)
        with gzip.open(os.path.join(path, 'knee_x_val.gz'), 'rb') as i:
            x_valid = pickle.load(i)
        with gzip.open(os.path.join(path, 'knee_x_test.gz'), 'rb') as i:
            x_test = pickle.load(i)
        with gzip.open(os.path.join(path, 'knee_y_train.gz'), 'rb') as i:
            y_train = pickle.load(i)
        with gzip.open(os.path.join(path, 'knee_y_val.gz'), 'rb') as i:
            y_valid = pickle.load(i)
        with gzip.open(os.path.join(path, 'knee_y_test.gz'), 'rb') as i:
            y_test = pickle.load(i)

    train_dataset = DPDataset(x_train, y_train, transform=None, dataset='chest')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = DPDataset(x_valid, y_valid, transform=None, dataset='chest')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = DPDataset(x_test, y_test, transform=None, dataset='chest')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


class NoisyDataset(Dataset):
    """Dataset with targets predicted by ensemble of teachers.
       Args:
            dataloader (torch dataloader): The original torch dataloader.
            model(torch model): Teacher model to make predictions.
            transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, dataloader, predictionfn, transform=None):
        self.dataloader = dataloader
        self.predictionfn = predictionfn
        self.transform = transform
        self.noisy_data = self.process_data()

    def process_data(self):
        """
        Replaces original targets with targets predicted by ensemble of teachers.
        Returns:
            noisy_data[torch tensor]: Dataset with labels predicted by teachers

        """

        noisy_data = []

        for data, _ in self.dataloader:
            noisy_data.append([data, torch.tensor(self.predictionfn(data)["predictions"])])

        return noisy_data

    def __len__(self):
        return len(self.dataloader)

    def __getitem__(self, idx):

        sample = self.noisy_data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


class DPDataset(Dataset):
    def __init__(self, x, y, transform=None, dataset='chest'):
        self.x = x
        self.y = y
        self.transform = transform
        self.dataset = dataset

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.x[idx]
        image = transforms.ToTensor()(image)
        if self.transform:
            image = self.transform(image)
        image = transforms.Resize((64, 64))(image)
        label = torch.zeros(2) if self.dataset == 'chest' else torch.zeros(5)
        label[self.y[idx]] = 1
        return (image, label)