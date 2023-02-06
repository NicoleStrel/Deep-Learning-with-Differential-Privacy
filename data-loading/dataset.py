# sample image data augmentations
import torch 
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

class DPDataset(Dataset):
    """
    Custom dataset for loading images
    """

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = self.data[idx]

        # normalize the pixel values to between 0 and 1 and crop to same size 
        # for DataLoader to work
        image = transforms.ToTensor()(image)
        if self.transform:
            image = self.transform(image)
        image = transforms.Resize((224,224))(image)
        return image

def data_loader(data, batch_size, shuffle=True, num_workers=0):
    """
    Custom data loader
    """
    
    dataset = DPDataset(data, transform=transform)

    # add gaussian noise to the images
    class gaussian_noise(object):
        def __init__(self, mean=0, std=0.5):
            self.std = std
            self.mean = mean

        def __call__(self, image):
            return image + torch.randn_like(image) * self.std + self.mean

    transform = gaussian_noise()
    dataset = ConcatDataset([dataset, DPDataset(data, transform=transform)])

    # add rotation of 90 degrees to the images
    transform = transforms.RandomRotation(degrees=(90,90))
    dataset = ConcatDataset([dataset, DPDataset(data, transform=transform)])

    # add random crop to the images
    transform = transforms.RandomResizedCrop(224)
    dataset = ConcatDataset([dataset, DPDataset(data, transform=transform)])

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, \
        num_workers=num_workers)
    return loader