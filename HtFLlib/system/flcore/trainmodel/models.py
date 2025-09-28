import math
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor

from flcore.trainmodel.mobilenet_v2 import *


class BaseHeadSplit(nn.Module):
    def __init__(self, args, cid):
        super().__init__()

        self.base = eval(args.models[cid % len(args.models)])
        head = None  # you may need more code for pre-existing heterogeneous heads
        if hasattr(self.base, 'heads'):
            head = self.base.heads
            self.base.heads= nn.AdaptiveAvgPool1d(args.feature_dim)
            
        elif hasattr(self.base, 'fcmbv2'):
            head = self.base.fcmbv2
            channel= self.base.fcmbv2.in_features
            self.base.fcmbv2 = nn.AdaptiveAvgPool1d(args.feature_dim)
        
        elif hasattr(self.base, 'fc'):
            head = self.base.fc
            channel= self.base.fc.in_features
            self.base.fc = nn.AdaptiveAvgPool1d(args.feature_dim)

              
        elif hasattr(self.base, 'fcfacnn'):
            head = self.base.fcfacnn
            channel= self.base.fcfacnn.in_features
            self.base.fcfacnn = nn.AdaptiveAvgPool1d(args.feature_dim)

        else:
            raise ('The base model does not have a classification head.')

        if hasattr(args, 'heads'):
            self.head = eval(args.heads[cid % len(args.heads)])
        elif 'vit' in args.models[cid % len(args.models)]:
            self.head = nn.Sequential(
                nn.Linear(args.feature_dim, 10), 
                nn.Tanh(),
                nn.Linear(10, args.num_classes)
            )
        else:
            self.head = nn.Linear(512, args.num_classes)
        
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out

class Head(nn.Module):
    def __init__(self, num_classes=10, hidden_dims=[512]):
        super().__init__()
        hidden_dims.append(num_classes)

        layers = []
        for idx in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[idx-1], hidden_dims[idx]))
            layers.append(nn.ReLU(inplace=True))
        self.fc = nn.Sequential(*layers)
    def forward(self, rep):
        out = self.fc(rep)
        return out

class FedAvgCNN(nn.Module):
    def __init__(self, in_features=1, num_classes=10, dim=1024):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512), 
            nn.ReLU(inplace=True)
        )
        self.fcfacnn = nn.Linear(512, num_classes)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.fcfacnn(out)
        return out

