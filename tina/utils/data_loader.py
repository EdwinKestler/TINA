# tina/utils/data_loader.py

import torch
from torchvision import datasets, transforms

def load_data(batch_size=64, data_dir='./data'):
    """
    Loads and returns training and test datasets for MNIST (or similar datasets).

    Args:
        batch_size (int): The batch size for data loading.
        data_dir (str): The directory to store/load the dataset.

    Returns:
        train_loader, test_loader: PyTorch data loaders for training and test data.
    """
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
    testset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
