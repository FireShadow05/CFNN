import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import softmax


def dataset(dataset = 'cifar-10',tf = transforms.ToTensor(),tf_test = transforms.ToTensor(),dir = '.',batch_size = 128):
    if dataset == 'cifar-10':
        train_dataset = torchvision.datasets.CIFAR10(root=dir,
                                                    train=True, 
                                                    transform=tf,
                                                    download=True)

        test_dataset = torchvision.datasets.CIFAR10(root=dir,
                                                    train=False, 
                                                    transform=tf_test)


        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=False)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size, 
                                                shuffle=False)

    if dataset == 'cifar-100':
        train_dataset = torchvision.datasets.CIFAR100(root=dir,
                                                    train=True, 
                                                    transform=tf,
                                                    download=True)

        test_dataset = torchvision.datasets.CIFAR100(root=dir,
                                                    train=False, 
                                                    transform=tf_test)


        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size, 
                                                shuffle=False)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=batch_size, 
                                                shuffle=False)
    
    return train_loader,test_loader