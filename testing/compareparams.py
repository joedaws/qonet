#!/usr/bin/python3
"""
file: compareparams.py

Here we compare the parameterizations of x^2 either by scaling
the input or by using a network which can approximatoin x^2 on [a,b]
"""
import sys
sys.path.append("../")
import matplotlib.pyplot as plt
import torch
import numpy as np
# import the networks in question
from polyinit import *

def compare_bitree(plotting=True):
    """
    Compares if sequential applications of products cause
    issues between the two parameterizations
    """
    in_N = 8
    num_L = 5
    min_val = -2
    max_val = 2
    
    # BiProdTree
    net1 = BiTreeProd(in_N,num_L,min_val,max_val)
    net1.poly_init()

    # BiProdTree_scale
    M = 2.1
    net2 = BiTreeProd_scale(in_N,num_L,M)
    net2.poly_init()

    # generate 50 random 8d points on the interval [-2,2]
    x = torch.Tensor(size=(50,8))
    x.uniform_(-2,2)

    # find truth
    y_truth = x[:,0:1]
    for i in range(1,8):
        y_truth = torch.mul(y_truth,(x[:,i:i+1]))

    # evaluate each network
    y1 = net1(x)
    y2 = net2(x)

    # plot the performance
    if plotting == True:
        fig, ax = plt.subplots()
        ax.scatter(y_truth.detach().numpy(),y_truth.detach().numpy(),label='truth')
        ax.scatter(y_truth.detach().numpy(),y1.detach().numpy(),label='interval adjust')
        ax.scatter(y_truth.detach().numpy(),y2.detach().numpy(),label='scaled adjust')
        ax.legend()
        plt.show()

    Nt = 50 # number of triails
    interval_error = 0
    scaled_error = 0
    # find the performance
    for i in range(0,Nt):
        # generate 50 random 8d points on the interval [-2,2]
        x = torch.Tensor(size=(50,8))
        x.uniform_(-2,2)

        # find truth
        y_truth = x[:,0:1]
        for j in range(1,8):
            y_truth = torch.mul(y_truth,(x[:,j:j+1]))

        # evaluate each network
        y1 = net1(x)
        y2 = net2(x)

        yy = y_truth.detach().numpy()
        yy1 = y1.detach().numpy()
        yy2 = y2.detach().numpy()
        interval_error += np.linalg.norm(abs(yy-yy1))/Nt
        scaled_error += np.linalg.norm(abs(yy-yy2))/Nt

    print("After %d trials:"%(Nt))
    print("Error of interval parameterization: %e"%interval_error)
    print("Error of scaled parameterization: %e"%scaled_error)

    return 

if __name__ == "__main__":
    # perform comparison of parameterizations
    compare_bitree()
