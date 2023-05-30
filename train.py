import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import softmax
import torch.nn.init as init
from models import *

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.nn import NLLLoss
from data import dataset

normalize_transform = transforms.Normalize(
        mean=torch.tensor([125.29552899, 122.99125831, 113.90624687]) / 256,
        std=torch.tensor([62.9836127, 62.04402182, 66.63918649]) / 256
    )
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    normalize_transform])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_transform])

train_loader,test_loader = dataset('cifar-10',transform,transform_test)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

lr = 0.001
momentum = 0.9
weight_decay = 0.005
num_epochs = 200
num_samples = 25

#Training of the model
def Training(model,samples = None):
    Loss = NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,momentum = 0.9,weight_decay=0.005)
    scheduler = CosineAnnealingWarmRestarts(optimizer,T_0 = num_epochs,eta_min = 0)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum = 0.9, weight_decay = 0.005)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            if samples:
                output = model(images,samples)
                loss = Loss(output, labels) 
            else:
                output = model(images) 
            loss = Loss(torch.log(output+1e-015), labels)

            # Backward and optimize
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print (" Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        scheduler.step()

def Testing(model,samples=None):
    model.eval()
    mean_accuracy = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            if samples:
                output = model(images, samples) 
            else:
                output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    
    
    return 100*correct/total

baseline_resnet = DeterministicNetwork(ResNet6(), 10).to(device)

Training(baseline_resnet)


