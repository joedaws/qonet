#!/usr/bin/python3
"""
File:
    prodnet.py

Author(s): 
    Joseph Daws Jr
Last Modified: 
    August 16, 2019

Description: 
    + ProdNet: 
      - a network with linear layers that computes the
        multiplication of its inputs
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.linalg import block_diag 
import matplotlib.pyplot as plt

class BiTreeProd(nn.Module):
    def __init__(self,in_N,num_L,min_val,max_val):
        """
        computes the product of all inputs
        """
        super(BiTreeProd,self).__init__()

        self.in_N = in_N
        self.num_L = num_L
        self.max_val = max_val
        self.min_val = min_val

        # list of prodnets
        self.hidden = nn.ModuleList()

        # number of prodnets required

        # case when in_N is 1
        if in_N == 1:
            self.hidden.append(ProdNet(in_N,num_L,min_val,max_val))
        
        # case for in_N more than 1
        elif in_N > 1:
            # addpend prodnets to the list
            this_L = num_L
            self.hidden.append(ProdNet(in_N,num_L,min_val,max_val))
            current_N = self.hidden[0].out_N
            i = 1
            
            # append the rest of the prodnets
            while current_N > 1:
                a = min_val
                b = max_val
                delta = 3*(b-a)**2/(2**(2*self.num_L+2))
                new_min = min(a*b,a*a,b*b) - delta
                new_max = max(a*b,a*a,b*b) + delta
                self.hidden.append(ProdNet(current_N,num_L,new_min,new_max))
                
                # update number of remaining items to multiply
                current_N = self.hidden[i].out_N
                # increment depth counter
                i += 1
                # update min and max
                min_val = new_min
                max_val = new_max

        # count number of prodnet used
        self.num_prod = len(self.hidden)

    def forward(self,x):
        """
        input should be (N,in_N) where N is number of samples
        output should be (N,1)
        """
        # first hidden layer
        h = self.hidden[0](x)
        
        # testing
        for i in range(1,self.num_prod):
            h = self.hidden[i](h)

        return h
    
    def poly_init(self):
        """
        initializes each sub-prodnet according to our polynomials
        """
        with torch.no_grad():
            for i in range(0,len(self.hidden)):
                self.hidden[i].poly_init()

        return

    def set_double(self):
        for h in self.hidden:
            h.set_double()

class ProdNet(nn.Module):
    def __init__(self,in_N,num_L,a,b):
        """
        computes the pairwise products of the inputs
        """
        super(ProdNet,self).__init__()
        
        # module list
        self.hidden = nn.ModuleList()
        
        # number of hidden units on middle layers
        # in_N even
        if in_N % 2 == 0: 
            mid_N = int(6*in_N)
            out_N = int(in_N/2)
        # odd case
        if in_N % 2 != 0:
            mid_N = int(6*(in_N-1)) + 2
            out_N = int(in_N/2) + 1

        self.in_N = in_N
        self.out_N = out_N
        self.num_L = num_L
        self.mid_N = mid_N
        self.out_N = out_N
        self.a = a
        self.b = b
        self.C = (b-a)**2/4

        # first hidden layer
        self.hidden.append(nn.Linear(in_N,mid_N))

        # iterate over middle hidden layers
        for i in range(1,self.num_L):
            self.hidden.append(nn.Linear(mid_N,mid_N))

        # output
        self.hidden.append(nn.Linear(mid_N,out_N))

    def forward(self,x):
        """
        x -- input of size (N,in_N) where N is the number of 
             sample points.
        """
        # first hidden layer
        h = self.hidden[0](x).clamp(min=0)

        # middle layers
        for i in range(1,self.num_L):
            h = self.hidden[i](h).clamp(min=0)

        # output layer
        return self.hidden[-1](h)

    def poly_init(self):
        """
        Initializes all network parameters so that it behaves like a 
        polynomial on the domain [a,b]^d
        """
        with torch.no_grad():
            a = self.a
            b = self.b
            C = self.C
            in_N = self.in_N
            num_L = self.num_L
            mid_N = self.mid_N
            
            # EVEN case
            if self.in_N % 2 == 0:
                A1 = [[1,0],
                      [1,0],
                      [1,0],
                      [a+b,0],
                      [0.5,0.5],
                      [0.5,0.5],
                      [0.5,0.5],
                      [(a+b)/2,(a+b)/2],
                      [0,1],
                      [0,1],
                      [0,1],
                      [0,a+b]]

                # input --> first hidden layer
                self.hidden[0].weight.data.fill_(0)
                for i in range(0,int(in_N/2)):
                    self.hidden[0].weight.data[12*i:(12*(i+1)),2*i:2*(i+1)] = \
                    torch.tensor(A1,dtype=torch.float)
                
                for i in range(0,int(mid_N/4)):
                    self.hidden[0].bias.data[4*i:4*(i+1)] = \
                    torch.tensor([-a,-(a+b)*0.5,-b,-a*b],dtype=torch.float)
                
                # first hidden --> second hidden
                A12 = [[2/(b-a),-4/(b-a),2/(b-a),0.0],
                       [2/(b-a),-4/(b-a),2/(b-a),0.0],
                       [2/(b-a),-4/(b-a),2/(b-a),0.0],
                       [-C*(2/(b-a)),C*(4/(b-a)),-C*(2/(b-a)),1.0]]
                
                self.hidden[1].weight.data.fill_(0)
                for i in range(0,int(mid_N/4)):
                    self.hidden[1].weight.data[4*i:4*(i+1),4*i:4*(i+1)] = \
                    torch.tensor(A12,dtype=torch.float)
                
                for i in range(0,int(mid_N/4)):
                    self.hidden[1].bias.data[4*i:4*(i+1)] = \
                    torch.tensor([0.,-0.5,-1.0,0.0],dtype=torch.float)
                
                # hk --> hk+1
                for k in range(2,num_L):
                    hk = self.hidden[k]
                    hk.weight.data.fill_(0)
                    A2 = [[2,-4,2,0.0],
                          [2,-4,2,0.0],
                          [2,-4,2,0.0],
                          [-2*C/(2**(2*(k-1))),
                            4*C/(2**(2*(k-1))),
                           -2*C/(2**(2*(k-1))),
                            1.0]]
        
                    for i in range(0,int(mid_N/4)):
                        hk.weight.data[i*4:(i+1)*4,i*4:(i+1)*4] = \
                        torch.tensor(A2,dtype=torch.float)
                    
                        hk.bias.data[4*i:4*(i+1)] = \
                        torch.tensor([0,-0.5,-1,0],dtype=torch.float)

                # output layer
                ii = num_L-1
                self.hidden[-1].bias.data.fill_(0.)
                self.hidden[-1].weight.data.fill_(0.)

                A3 = [ C/(2**(2*ii)),
                      -C/(2**(2*ii)),
                       C/(2**(2*ii)),
                      -0.5,
                      -2*2*C/(2**(2*ii)),
                       2*4*C/(2**(2*ii)),
                      -2*2*C/(2**(2*ii)),
                       2*1.0,
                       C/(2**(2*ii)),
                      -C/(2**(2*ii)),
                       C/(2**(2*ii)),
                      -0.5]
                for i in range(0,int(in_N/2)): 
                    self.hidden[-1].weight.data[i,12*i:12*(i+1)] = \
                    torch.tensor(A3,dtype = torch.float) 
            
            # ODD case
            if self.in_N % 2 != 0:
                A1 = [[1,0],
                      [1,0],
                      [1,0],
                      [a+b,0],
                      [0.5,0.5],
                      [0.5,0.5],
                      [0.5,0.5],
                      [(a+b)/2,(a+b)/2],
                      [0,1],
                      [0,1],
                      [0,1],
                      [0,a+b]]

                # input --> first hidden layer
                self.hidden[0].weight.data.fill_(0)
                # parameters for pairs
                for i in range(0,int(in_N/2)):
                    self.hidden[0].weight.data[12*i:(12*(i+1)),2*i:2*(i+1)] = \
                    torch.tensor(A1,dtype=torch.float)
                # parameters for single
                self.hidden[0].weight[-2,-1] = torch.tensor(1,dtype=torch.float)
                self.hidden[0].weight[-1,-1] = torch.tensor(-1,dtype=torch.float)
                
                # set biases
                # set all nodes to zero bias, this will take care of 
                # single nodes
                self.hidden[0].bias.data.fill_(0)
                # iterate over the multiplication nodes
                for i in range(0,int(mid_N/4)):
                    self.hidden[0].bias.data[4*i:4*(i+1)] = \
                    torch.tensor([-a,-(a+b)*0.5,-b,-a*b],dtype=torch.float)

                # FIRST hidden --> second hidden
                A12 = [[2/(b-a),-4/(b-a),2/(b-a),0.0],
                       [2/(b-a),-4/(b-a),2/(b-a),0.0],
                       [2/(b-a),-4/(b-a),2/(b-a),0.0],
                       [-C*(2/(b-a)),C*(4/(b-a)),-C*(2/(b-a)),1.0]]
                
                # set all weights to zero
                self.hidden[1].weight.data.fill_(0)
                # iterate over multiplication blocks
                for i in range(0,int(mid_N/4)):
                    self.hidden[1].weight.data[4*i:4*(i+1),4*i:4*(i+1)] = \
                    torch.tensor(A12,dtype=torch.float)
                # parameters for single
                self.hidden[1].weight.data[-2,-2] = 1
                self.hidden[1].weight.data[-1,-1] = 1
                
                # set biases
                self.hidden[1].bias.data.fill_(0)
                for i in range(0,int(mid_N/4)):
                    self.hidden[1].bias.data[4*i:4*(i+1)] = \
                    torch.tensor([0.,-0.5,-1.0,0.0],dtype=torch.float)
                
                # MIDDLE layers hk --> hk+1
                for k in range(2,num_L):
                    hk = self.hidden[k]
                    # set all weights to 0
                    hk.weight.data.fill_(0)
                    hk.bias.data.fill_(0)
                    # define multiplication parameters
                    A2 = [[2,-4,2,0.0],
                          [2,-4,2,0.0],
                          [2,-4,2,0.0],
                          [-2*C/(2**(2*(k-1))),
                            4*C/(2**(2*(k-1))),
                           -2*C/(2**(2*(k-1))),
                            1.0]]
                    # iterate over multiplication blocks
                    for i in range(0,int(mid_N/4)):
                        hk.weight.data[i*4:(i+1)*4,i*4:(i+1)*4] = \
                        torch.tensor(A2,dtype=torch.float)
                    
                        hk.bias.data[4*i:4*(i+1)] = \
                        torch.tensor([0,-0.5,-1,0],dtype=torch.float)
                    
                    # middle layer single parameters
                    hk.weight.data[-2,-2] = 1
                    hk.weight.data[-1,-1] = 1
                

                # OUTPUT layer
                ii = num_L-1
                self.hidden[-1].bias.data.fill_(0.)
                self.hidden[-1].weight.data.fill_(0.)
                # define multiplication parameters
                A3 = [ C/(2**(2*ii)),
                      -C/(2**(2*ii)),
                       C/(2**(2*ii)),
                      -0.5,
                      -2*2*C/(2**(2*ii)),
                       2*4*C/(2**(2*ii)),
                      -2*2*C/(2**(2*ii)),
                       2*1.0,
                       C/(2**(2*ii)),
                      -C/(2**(2*ii)),
                       C/(2**(2*ii)),
                      -0.5]
                # iterate over blocks
                for i in range(0,int(in_N/2)):
                    self.hidden[-1].weight.data[i,12*i:12*(i+1)] = \
                    torch.tensor(A3,dtype = torch.float) 
                
                # last layer single parameters
                self.hidden[-1].weight.data[-1,-2] = 1
                self.hidden[-1].weight.data[-1,-1] = -1

        return

    def xavier_init(self):
        """
        initializes the linear layers using Xavier initialization.
        The weights are initialized using xavier random initialization.
        The biases use uniform initialization on the interval of approximation.
        """
        with torch.no_grad():
            # iterate over the hidden layers
            for h in self.hidden:
                torch.nn.init.xavier_uniform_(h.weight)
                h.bias.uniform_(self.a,self.b)
    
    def set_double(self):
        for h in self.hidden:
            h.double()

class BiTreeProd_scale(nn.Module):
    def __init__(self,in_N,num_L,M):
        """
        computes the product of all inputs
        but scales appropriatly
        """
        super(BiTreeProd_scale,self).__init__()

        self.in_N = in_N
        self.num_L = num_L
        self.M = M

        # list of prodnets
        self.hidden = nn.ModuleList()

        # number of prodnets required

        # case when in_N is 1
        if in_N == 1:
            self.hidden.append(ProdNet(in_N,num_L,-2,2))
        # case for
        elif in_N > 1:
            # addpend prodnets to the list
            this_L = num_L
            self.hidden.append(ProdNet(in_N,num_L,-2,2))
            current_N = self.hidden[0].out_N
            i = 1
            # append the rest of the prodnets
            while current_N > 1:
                self.hidden.append(ProdNet(current_N,num_L,-2,2))
                # update number of remaining items to multiply
                current_N = self.hidden[i].out_N
                # increment depth counter
                i += 1

        # count number of prodnet used
        self.num_prod = len(self.hidden)

    def forward(self,x):
        """
        input should be (N,in_N) where N is number of samples
        output should be (N,1)
        """
        M = self.M

        # first hidden layer
        h = M**2*self.hidden[0](x/M)
        
        for i in range(1,self.num_prod):
            delta = 3*M**2/(2**(2*self.num_L+2))
            M = M**2+delta
            h = M**2*self.hidden[i](h/M)

        return h
    
    def poly_init(self):
        """
        initializes each sub-prodnet according to our polynomials
        """
        with torch.no_grad():
            for i in range(0,len(self.hidden)):
                self.hidden[i].poly_init()

        return

def test():
    print("TESTING EVEN CASE")
    # plot a test for the case when in_N = 2
    net = ProdNet(in_N=2,num_L=5,a=0,b=1)
    net.poly_init()
    x = torch.rand(100,2)
    y_truth = torch.mul(x[:,0:1],x[:,1:2])
    y_pred = net(x)
   
    """
    fig, ax = plt.subplots()
    ax.scatter(y_truth.detach().numpy(),y_truth.detach().numpy(),label='truth')
    ax.scatter(y_truth.detach().numpy(),y_pred.detach().numpy(),label='predicted')
    ax.legend()
    plt.show()
    """
    print("\n\n")

    print("TESTING ODD CASE")
    net2 = ProdNet(in_N=5,num_L=9,a=-3,b=2)
    net2.poly_init()
    print(net2.mid_N)
    print(net2.out_N)
    x2 = torch.rand(1,5)
    y2_truth = [x2[0,0]*x2[0,1],x2[0,2]*x2[0,3],x2[0,4]]
    y2_pred = net2(x2)
    print("Truth")
    print(y2_truth)
    print("Prediction")
    print(y2_pred)
    
    print("\n\n")
    
    print("TESTING BINARY PRODUCT")
    # testing BiTreeProd
    net3 = BiTreeProd(in_N=5,num_L=5,min_val=-2.1,max_val=2.1)
    net3.poly_init()
    x = torch.Tensor(size=(50,5))
    x.uniform_(-2,2)
    y3_pred = net3(x)
    y3_truth = x[:,0:1]*x[:,1:2]*x[:,2:3]*x[:,3:4]*x[:,4:5]
   
    print(y3_truth.size())
    print(y3_pred.size())

    fig, ax = plt.subplots()
    ax.scatter(y3_truth.detach().numpy(),y3_truth.detach().numpy(),label='truth')
    ax.scatter(y3_truth.detach().numpy(),y3_pred.detach().numpy(),label='predicted')
    ax.legend()
    plt.show()

# perform the test
if (__name__ == "__main__"):
    test()
