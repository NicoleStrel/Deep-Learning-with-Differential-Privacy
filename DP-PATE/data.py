# from torchvision import datasets, transforms
from torch.utils.data import Dataset

# from torch.utils.data.sampler import SubsetRandomSampler
# import numpy as np

# import copy
import torch
from torch.utils.data import Dataset, TensorDataset

import pickle
import gzip
import os

import numpy


def load_data(train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset, batch_size: int):
    """Load train data, valid data, and test data"""
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader


def get_datasets(dataset: str):
    """Returns train, valid and test datasets"""
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

    train_set = create_dataset(x_train, y_train)
    valid_set = create_dataset(x_valid, y_valid)
    test_set = create_dataset(x_test, y_test)

    return train_set, valid_set, test_set


def create_dataset(x_array, y_array):
    """Helper function for get_datasets() function"""
    # x = torch.from_numpy(x_array).view(x_array.shape[0], -1)  # convert 4d array to 2d
    # x = torch.from_numpy(x_array)

    # x_array in the form (N, h, w, 3), need it to be (N, 3, h, w)
    x = torch.from_numpy(x_array).view(x_array.shape[0], x_array.shape[3], x_array.shape[1], x_array.shape[2])
    y = torch.from_numpy(y_array.flatten()).long()

    return TensorDataset(x.clone().detach(), y.clone().detach())
    # return TensorDataset(torch.tensor(x_array), torch.tensor(y_array))


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





