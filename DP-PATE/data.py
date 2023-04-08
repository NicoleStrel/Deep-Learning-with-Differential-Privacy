import torch
from torch.utils.data import Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import pickle
import gzip
import os


def load_data(train_dataset_t: Dataset, train_dataset_s: Dataset, test_dataset: Dataset, batch_size: int):
    """
    Load train data for teachers, train data for student, and test data
    """
    # t = teacher; s = student
    train_loader_t = torch.utils.data.DataLoader(train_dataset_t, batch_size=batch_size, shuffle=True)
    train_loader_s = torch.utils.data.DataLoader(train_dataset_s, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader_t, train_loader_s, test_loader


def get_datasets(dataset: str):
    """
    Returns train dataset for teachers, train dataset for students, and test datasets
    """
    if dataset == 'chest':
        path = 'chest-data'  # os.path.join(os.path.dirname(os.getcwd()), 'chest-data')
        with gzip.open(os.path.join(path, 'chest_x_train.gz'), 'rb') as i:
            x_train = pickle.load(i)
        # with gzip.open(os.path.join(path, 'chest_x_val.gz'), 'rb') as i:
        #     x_valid = pickle.load(i)
        with gzip.open(os.path.join(path, 'chest_x_test.gz'), 'rb') as i:
            x_test = pickle.load(i)
        with gzip.open(os.path.join(path, 'chest_y_train.gz'), 'rb') as i:
            y_train = pickle.load(i)
        # with gzip.open(os.path.join(path, 'chest_y_val.gz'), 'rb') as i:
        #     y_valid = pickle.load(i)
        with gzip.open(os.path.join(path, 'chest_y_test.gz'), 'rb') as i:
            y_test = pickle.load(i)

    else:
        path = 'knee-data'  # os.path.join(os.path.dirname(os.getcwd()), 'knee-data')
        with gzip.open(os.path.join(path, 'knee_x_train.gz'), 'rb') as i:
            x_train = pickle.load(i)
        # with gzip.open(os.path.join(path, 'knee_x_val.gz'), 'rb') as i:
        #     x_valid = pickle.load(i)
        with gzip.open(os.path.join(path, 'knee_x_test.gz'), 'rb') as i:
            x_test = pickle.load(i)
        with gzip.open(os.path.join(path, 'knee_y_train.gz'), 'rb') as i:
            y_train = pickle.load(i)
        # with gzip.open(os.path.join(path, 'knee_y_val.gz'), 'rb') as i:
        #     y_valid = pickle.load(i)
        with gzip.open(os.path.join(path, 'knee_y_test.gz'), 'rb') as i:
            y_test = pickle.load(i)

    x_train_1, x_train_2, y_train_1, y_train_2 = train_test_split(x_train, y_train, test_size=0.20, random_state=42)

    train_teacher = create_dataset(x_train_1, y_train_1)  # assume this data is sensitive
    train_student = create_dataset(x_train_2, y_train_2)  # assume this data has no labels (differentially private)
    test_set = create_dataset(x_test, y_test)

    return train_teacher, train_student, test_set


def create_dataset(x_array, y_array):
    """
    Helper function of get_datasets() function.
    Convert numpy arrays x_array and y_array into Tensor Datasets
    """
    # x_array in the form (N, h, w, 3), need it to be (N, 3, h, w)
    x = torch.from_numpy(x_array / 255).view(x_array.shape[0], x_array.shape[3], x_array.shape[1], x_array.shape[2])
    y = torch.from_numpy(y_array.flatten()).long()

    return TensorDataset(x.clone().detach(), y.clone().detach())


class NoisyDataset(Dataset):
    """Dataset with targets predicted by ensemble of teachers.
       Args:
            dataloader (torch dataloader): The original torch dataloader.
            predictionfn (callable): The teacher's prediction function
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
            noisy_data (torch tensor): Dataset with labels predicted by teachers

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
