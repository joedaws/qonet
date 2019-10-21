"""
ResNet for d-dimensional functions functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self,in_d=1,out_d=1,width=10,depth=10):
        super(ResNet, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.width = width
        self.depth = depth
        self.sigma = F.relu
        self.layer = nn.ModuleList()

        # first hidden layer
        self.layer.append(nn.Linear(in_d,width))

        # middle layers
        for i in range(1,depth-1):
            self.layer.append(nn.Linear(width,width))

        # last hidden layer
        self.layer.append(nn.Linear(width,out_d))

    def forward(self, x):
        x = self.sigma(self.layer[0](x))
        
        for i in range(1,depth-1):
            x = x + self.sigma(self.layer[i](x))

        x = self.layer[self.depth](x)
        
        return x

def test():

    net = ResNet()

    return

if __name__ == "__main__":
    test()
