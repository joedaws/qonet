"""
ResNet for 1d functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet1d(nn.Module):

    def __init__(self, width):
        super(ResNet1d, self).__init__()
        self.width = width
        self.h = 1
        self.sigma = F.relu
        
        self.layer1 = nn.Linear(1,width)
        self.layer2 = nn.Linear(width,width)
        self.layer3 = nn.Linear(width,width)
        self.layer4 = nn.Linear(width,width)
        self.layer5 = nn.Linear(width,width)
        self.layer6 = nn.Linear(width,width)
        self.layer7 = nn.Linear(width,width)
        self.layer8 = nn.Linear(width,width)
        self.layer9 = nn.Linear(width,width)
        self.layer10 = nn.Linear(width,width)
        self.layer11 = nn.Linear(width,width)
        self.layer12 = nn.Linear(width,width)
        self.layer13 = nn.Linear(width,width)
        self.layer14 = nn.Linear(width,width)
        self.layer15 = nn.Linear(width,width)
        self.layer16 = nn.Linear(width,width)
        self.layer17 = nn.Linear(width,width)
        self.layer18 = nn.Linear(width,width)
        self.layer19 = nn.Linear(width,1)

    def forward(self, x):
        x = self.layer1(x)
        x = x + self.sigma(self.layer2(x))
        x = x + self.sigma(self.layer3(x))
        x = x + self.sigma(self.layer4(x))
        x = x + self.sigma(self.layer5(x))
        x = x + self.sigma(self.layer6(x))
        x = x + self.sigma(self.layer7(x))
        x = x + self.sigma(self.layer8(x))
        x = x + self.sigma(self.layer9(x))
        x = x + self.sigma(self.layer10(x))
        x = x + self.sigma(self.layer11(x))
        x = x + self.sigma(self.layer12(x))
        x = x + self.sigma(self.layer13(x))
        x = x + self.sigma(self.layer14(x))
        x = x + self.sigma(self.layer15(x))
        x = x + self.sigma(self.layer16(x))
        x = x + self.sigma(self.layer17(x))
        x = x + self.sigma(self.layer18(x))
        x = self.layer19(x)
        
        return x

