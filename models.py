import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import softmax

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels,track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels,track_running_stats=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, layers, block = ResidualBlock, num_classes=256):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16,track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes, bias = True)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels,track_running_stats=False))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

class ResNet2(nn.Module):
    def __init__(self, block = ResidualBlock, num_classes=256):
        super(ResNet2, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16,track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(32)
        self.fc = nn.Linear(16, num_classes, bias = True)
        self.apply(_weights_init)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels,track_running_stats=False))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

class ResNet6(nn.Module):
    def __init__(self, block = ResidualBlock, num_classes=256):
        super(ResNet6, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16,track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, 1)
        self.layer2 = self.make_layer(block, 32, 1, 2)
        self.avg_pool = nn.AvgPool2d(16)
        self.fc = nn.Linear(32, num_classes, bias = True)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels,track_running_stats=False))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
            out = self.conv(x)
            out = self.bn(out)
            out = self.relu(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

class ResNet10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet10, self).__init__()
        self.ResNet = ResNet([2,1,1])
    def forward(self,x):
        out = self.ResNet(x)
        return out

class ResNet14(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet14, self).__init__()
        self.ResNet = ResNet([2,2,2])
    def forward(self,x):
        out = self.ResNet(x)
        return out

class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        self.ResNet = ResNet([3,3,3])
    def forward(self,x):
        out = self.ResNet(x)
        return out

class ResNet56(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet56, self).__init__()
        self.ResNet = ResNet([9,9,9])
    def forward(self,x):
        out = self.ResNet(x)
        return out

class DeterministicNetwork(nn.Module):
    def __init__(self, base, num_classes=10):
        super(DeterministicNetwork, self).__init__()
        self.base = base
        self.fc = nn.Linear(256,num_classes, bias = True)
    def forward(self,x):
        out = self.base(x)
        out = self.fc(out)
        return softmax(out, dim = 1)
    
def conv3x31d(in_channels, out_channels, stride=1):
    return nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
    
class ResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1d, self).__init__()
        self.conv1 = conv3x31d(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm1d(out_channels,track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x31d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels,track_running_stats=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
class Ge(nn.Module):
    def __init__(self, block = ResidualBlock1d , num_classes=256):
        super(Ge, self).__init__()
        self.in_channels = 16
        self.conv = conv3x31d(2, 16)
        self.bn = nn.BatchNorm1d(16,track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, 1)
        self.layer2 = self.make_layer(block, 32, 1, 2)
        self.avg_pool = nn.AvgPool1d(128)
        self.fc = nn.Linear(32, num_classes, bias = True)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x31d(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm1d(out_channels,track_running_stats=False))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class Gh(nn.Module):
    def __init__(self, block = ResidualBlock1d , num_classes=10):
        super(Gh, self).__init__()
        self.in_channels = 16
        self.conv = conv3x31d(3, 16)
        self.bn = nn.BatchNorm1d(16,track_running_stats=False)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, 1)
        self.layer2 = self.make_layer(block, 32, 1, 2)
        self.avg_pool = nn.AvgPool1d(128)
        self.fc1 = nn.Linear(32, 256*num_classes, bias = True)
        self.fc2 = nn.Linear(32,num_classes, bias = True)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x31d(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm1d(out_channels,track_running_stats=False))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out1 = self.fc1(out)
        out2 = self.fc2(out)
        return out1, out2

class GeneratorNetwork(nn.Module):
    def __init__(self,num_classes=10):
        super(GeneratorNetwork, self).__init__()
        self.Ge = Ge()
        self.num_classes = num_classes
        self.Gh = Gh(num_classes=num_classes)
        self.fc = nn.Linear(num_classes,256,bias=True)
    def forward(self,y,z1,z2,num_samples = 1):
        y = self.fc(y).view(num_samples,1,256)
        temp = self.Ge(torch.cat((z1,y),axis = 1))
        temp = temp.unsqueeze(1).repeat(1, 1, 1)
        W,b = self.Gh(torch.cat((z2,temp,y),axis=1))
        return W.view(num_samples,256,self.num_classes),b.view(num_samples,1,self.num_classes)

class CFNN(nn.Module):
    def __init__(self, base,num_classes=10 ):
        super(CFNN, self).__init__()
        self.base = base
        self.generator = GeneratorNetwork(num_classes = num_classes)
        self.num_classes = num_classes
    
    def sampler(self, num_samples): 
        m = torch.distributions.categorical.Categorical(torch.zeros(self.num_classes) + 1/self.num_classes)
        x = m.sample([int(num_samples)])
        return nn.functional.one_hot(x,self.num_classes).type(torch.float32)

    def forward(self,x,num_samples = 1):
        out = self.base(x)
        W,b = self.generator(self.sampler(num_samples).to(device),torch.rand(num_samples,1,256).to(device),torch.rand(num_samples,1,256).to(device),num_samples)
        out = softmax(torch.matmul(out,W) - b, dim=2)
        out = torch.mean(out,axis = 0)
        return out