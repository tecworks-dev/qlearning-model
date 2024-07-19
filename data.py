import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split


def get_transforms(config):
    train_transform = transforms.Compose([
        transforms.Resize(config['resize']),
        transforms.RandomHorizontalFlip() if config['random_horizontal_flip'] else lambda x: x,
        transforms.RandomRotation(config['random_rotation']),
        transforms.ToTensor(),
        transforms.Normalize(config['normalize_mean'], config['normalize_std']),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(config['resize']),
        transforms.ToTensor(),
        transforms.Normalize(config['normalize_mean'], config['normalize_std']),
    ])

    return train_transform, test_transform


def get_dataloaders(config):
    train_transform, test_transform = get_transforms(config['augmentation'])

    trainset = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_len = int(len(trainset) * (1 - config['train']['val_split']))
    val_len = len(trainset) - train_len
    train_subset, val_subset = random_split(trainset, [train_len, val_len])

    trainloader = DataLoader(train_subset, batch_size=config['train']['batch_size'], shuffle=True)
    valloader = DataLoader(val_subset, batch_size=config['train']['batch_size'], shuffle=False)
    testloader = DataLoader(testset, batch_size=config['train']['batch_size'], shuffle=False)

    return trainloader, valloader, testloader
