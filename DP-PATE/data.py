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


def load_data(batch_size, dataset: str):
    """Helper function used to load the train/test data.
       Args:
           train[boolean]: Indicates whether its train/test data.
           batch_size[int]: Batch size
    """

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

        train_loader = help_load_data(x_train, y_train, batch_size=batch_size)
        valid_loader = help_load_data(x_valid, y_valid, batch_size=batch_size)
        test_loader = help_load_data(x_test, y_test, batch_size=batch_size)

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

        train_loader = help_load_data(x_train, y_train, batch_size=batch_size)
        valid_loader = help_load_data(x_valid, y_valid, batch_size=batch_size)
        test_loader = help_load_data(x_test, y_test, batch_size=batch_size)

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


def help_load_data(x_array, y_array, batch_size):
    x = torch.from_numpy(x_array).view(x_array.shape[0], -1)  # convert 4d array to 2d
    y = torch.from_numpy(y_array.flatten())

    dataset = TensorDataset(x.clone().detach(), y.clone().detach())
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

