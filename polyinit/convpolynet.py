#!/usr/bin/python3
"""

Creates CNN which can be initialized to behave like polynomials.
"""

import torch
import torch.nn as nn

def convtest():
    # make 2d layer
    in_channels = 1
    out_channels = 1
    kernel_size = 2
    L1 = nn.Conv2d(in_channels,out_channels,kernel_size)
    with torch.no_grad():
        L1.weight.data.fill_(1.)
        L1.bias.data.fill_(0.)
    # make sample input
    x = torch.ones(1,1,4,4)
    
    # try this out
    y = L1(x)

    print("output:\n",y)
    print("input:\n",x)
    print("weights:\n",L1.weight.data)

if __name__ == '__main__':
    convtest()
