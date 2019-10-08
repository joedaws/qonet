#!/usr/bin/python3
"""
File:
    polynet.py

Author(s): 
    Joseph Daws Jr
Last Modified: 
    Oct 8, 2019

Description: 
    + PolyNet: 
      A class which can be initialized to 
      be any polynomial. Requires a PolyClass
      object in order to be instantiated

    + PolyInfo: 
      A class thats contain information need to define 
      the polynomial associated with the network.
      - dim      -- Dimension of tensor product
      - polytype -- Orthonormal set of polynomials
      - idxset   -- index set assocaited with exapansion
                    in the given orthonomral system.
      - roots    -- Roots associated with the polynomials
                    used in the expansion
      - coefs    -- Coefficients of these polynomials 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.linalg import block_diag 
import matplotlib.pyplot as plt

# import legendre root getter
from .roots import *

# import deafult coefs
from .defaults import defaultcoefs, defaultidx

# legendre polynomial evaluation
from numpy.polynomial.legendre import legval

# import product network
from .prodnet import BiTreeProd

class PolyInfo:
    """
    Class for holding the info related to the polynomial
    which generates the network
    """
    def __init__(self,dim=4,outdim=1,L=15,polytype='leg',idxset=defaultidx,coefs=defaultcoefs):
        """
        Initializes the network:
        dim        -- dimension of the input of the polynomial
        outdim     -- dimension of the output of the polynomial
        polytype   -- type of polynomial used in the tensor product basis
        idxset     -- index set that determines the terms in the poly
        all_bias   -- list of arrays of roots necessary to create first hidden 
                      layer of TensorPolynomial networks
        all_weight -- list of weight matrices necessary to create first hidden
                      layer of TensorPolynomial networks
        idxcard    -- cardinality of the index set, i.e., how many 
                      terms in the polynomial
        L          -- depth of each polynomial block
        first_wid  -- width of the first linear layer whose biases
                      are the requires numbers (x_i - r_j)
        """
        self.dim = dim
        self.outdim = outdim
        self.polytype = polytype
        self.idxset = idxset
        self.first_wid = sum(sum(self.idxset))
        self.idxcard = self.idxset.shape[0]
        self.L = L
        self.all_scalfac = np.zeros((self.idxcard,self.outdim))
        self.all_bias = self.get_bias()
        self.all_weight = self.get_weight()
        self.coef_mat = np.multiply(self.all_scalfac,coefs)
        self.coef_mat = np.reshape(self.coef_mat,(self.outdim,self.idxcard))
    
    def get_bias(self):
        """
        generates a vector of roots to be used to initializing the first
        layer's biases in the network based on the polytype and index set.
        
        NOTES:
        + Taylor polynomial case is unstable
        """
        if self.polytype == 'tay':
            # all roots are zero
            roots = np.zeros(self.first_wid)

        elif self.polytype == 'leg':
            # load all necessary legendre roots
            lr = leg_roots_vec()

            # testing a version where I append to a list
            roots = []
            # loop over index set and set necessary roots
            for count,i in enumerate(self.idxset):
                # list for accumulating roots
                r = []
                scalfac = 1
                # loop over each polynomial in tensor product
                for p in i:
                    # reordering list
                    reorder = []
                    # even case
                    if p % 2 == 0:
                        for i in range(0,int(p/2)):
                            reorder.append(i)
                            reorder.append(p-(i+1))
                    # odd case
                    else:
                        for i in range(0,int(p/2)):
                            reorder.append(i)
                            reorder.append(p-(i+1))
                        reorder.append(int(p/2))

                    # set scaling factor
                    scalfac *= get_leg_scalfac(p)
                    # get legendre roots for degree p Legendre polynomial
                    p_roots = lr[p]
                    # append the scaling factor
                    #r.append(scalfac)
                    # appen the roots in non constant case
                    if p > 0:
                        # include roots as bias
                        for j in range(0,p_roots.size):
                            thisone = p_roots[j]
                            r.append(thisone)
                # include scaling factor
                self.all_scalfac[count,:] = scalfac
                # reorder list of roots
                r = [r[i] for i in reorder]
                # convert r from list to numpy array
                roots.append(np.asarray(r))

        return roots

    def get_weight(self):
        """
        generates a list of weight matrices to be used for initializing
        the first layer of the TensorPolynomial network. 
        
        NOTES:
        + Does not depend on the polytypes
        + All weights are 0 and 1
        """
        # get legendre roots for book keeping only
        lr = leg_roots_vec()

        # weights list
        weights = []

        # loop over index set
        for i,nu in enumerate(self.idxset):
            # get roots needed for polynomial i
            r = self.all_bias[i]
            # we will need a certain weight matrix
            w_N = r.size
            w = np.ones([int(w_N),int(self.dim)])
            
            """
            # loop over each polynomial in tensor product
            # counter for setting the 1 in the correct place
            ctr = 0
            for ii,p in enumerate(nu):
                ctr += 1
                # only need to set a 1 if poly is non-constant
                if p > 0:
                    # get leg roots for degree p polynomial for book-keeping
                    p_roots = lr[p]
                    # place 1's in the correct places
                    for j in range(0,p_roots.size):
                        w[ctr+j,ii] = 1.0
            """
            # append and convert w from list to numpy array
            weights.append(np.asarray(w))

        return weights

    def eval_poly(self,x):
        """
        evaluates the polnomial at input point x (dim)
        and returns the result y (outdim)
        
        NOTES:
        + SINGLE POINT EVALUATION
        + may want to consider extending to multi-point later
        + Only working for one-dimensional case
        """
        # set the output value
        y = np.zeros([self.outdim])

        # iterate over the terms
        for i,nu in enumerate(self.idxset):
            val = 1
            for oned in nu:
                c = np.zeros(oned+1)
                c[oned] = 1
                val *= legval(x,c)

            val *= self.coef_mat[i,0]
            y += val

        return y

# default polyinfo for generic PolyNet
default_info = PolyInfo()

class PolyNet(nn.Module):
    def __init__(self,pinfo=default_info):
        """
        INPUTS:

        pinfo -- relavent information for 
        """
        super(PolyNet,self).__init__()
        # pinfo is PolyInfo type class
        self.info = pinfo
        # list of networks
        self.terms = nn.ModuleList()
        
        # instantiate a TensorPolyNetwork for EACH term in the network
        for i,nu in enumerate(self.info.idxset): 
            bias_vec = self.info.all_bias[i]
            weight_mat = self.info.all_weight[i]
            # see if more than quadratic
            if len(bias_vec) > 1:
                self.terms.append(\
                TensorPolynomial(self.info.dim,bias_vec,weight_mat,self.info.L)\
                )
            else:
                self.terms.append(nn.Linear(pinfo.dim,pinfo.outdim))

        # OUTPUT layer
        #self.last = nn.Linear(self.info.idxcard,self.info.outdim)
        self.last = nn.Linear(self.info.idxcard,self.info.outdim,bias=False)

    def forward(self,x):
        """
        x -- input of size (N,d) where N is the number of 
             sample points.
        """
        # variable for collecting output of each 
        lastx = torch.Tensor(x.shape[0],self.info.idxcard) 

        # iterate over each term in the polynomial expansion
        for i,nu in enumerate(self.terms):
            lastx[:,i:i+1] = nu(x)

        return self.last(lastx)

    def poly_init(self):
        """
        Initializes all network parameters so that it behaves like a 
        polynomial on the domain [a,b]^d
        """
        with torch.no_grad():
            # initialize TensorPolynomial
            for i,nu in enumerate(self.terms):
                if nu.__class__.__name__ == 'TensorPolynomial':
                    # initialize TensorPolynomial
                    nu.poly_init()
                if nu.__class__.__name__ == 'Linear':
                    # linear case
                    if sum(self.info.idxset[i]) == 1:
                        nu.weight.fill_(1.)
                        nu.bias.fill_(0.)
                    # constant case:
                    elif sum(self.info.idxset[i]) == 0:
                        nu.weight.fill_(0.)
                        nu.bias.fill_(1.)

            # initialize the coefficients in the last layer
            #self.last.bias.fill_(0.) 
            self.last.weight.data = \
            torch.tensor(self.info.coef_mat,dtype=torch.float)
        return

    def xavier_init(self):
        """
        initializes the linear layers using Xavier initialization.
        The weights are initialized using xavier random initialization.
        """
        with torch.no_grad():
            pass
        return 

class TensorPolynomial(nn.Module):
    def __init__(self,dim,bias_vec,weight_mat,num_L):
        """
        INPUTS:
        
        bias_vec -- np array of roots be used in the first layer
        weight_vec -- np array of weights to be used in the first layer
        num_L      -- number of layers to use in each ProdNet in the BiTreeNet
        """
        super(TensorPolynomial,self).__init__()
        
        self.dim = dim
        self.bias_vec = bias_vec
        self.weight_mat = weight_mat
        self.num_L = num_L
        self.max_val = max(bias_vec)
        self.min_val = min(bias_vec)

        # set some paramters for the BiTreeProd Network we will use 
        in_N = bias_vec.size

        # define first layer to transform inputs into factored polynomial type
        self.first = nn.Linear(dim,in_N)

        # BiTreeProd network for multiplying all necessary (x_i - r_i)
        self.btp = BiTreeProd(in_N,num_L,self.min_val,self.max_val)

    def forward(self,x):
        """
        forward propogation through the network
        """
        h = self.first(x)
        return self.btp(h)
        
    def poly_init(self):
        with torch.no_grad():
            # initialize first layer with roots and weights
            self.first.weight.data = \
            torch.tensor(self.weight_mat,dtype=torch.float)
            self.first.bias.data = \
            torch.tensor(self.bias_vec,dtype=torch.float)

            # polyinit the BiTreeProd Network
            self.btp.poly_init()

        return

    def set_double(self):
        # set first layer to double
        self.first.double()

        # set btp to double
        self.btp.set_double()

        return
"""
# a default index set of dim=4
# this is a total degree set of order 2
defaultidx = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 2],
                       [0, 0, 1, 1],
                       [0, 1, 0, 1],
                       [1, 0, 0, 1],
                       [0, 0, 2, 0],
                       [0, 1, 1, 0],
                       [1, 0, 1, 0],
                       [0, 2, 0, 0],
                       [1, 1, 0, 0],
                       [2, 0, 0, 0]])

# a default set of coefficients
defaultcoefs = np.array([[ 0.2],
                         [-0.4],
                         [-0.1],
                         [ 0.2],
                         [ 0.5],
                         [ 0.3],
                         [ 0.1],
                         [ 0.3],
                         [-0.6],
                         [ 0.4],
                         [ 0.3],
                         [-0.7],
                         [ 0.3],
                         [ 0.4],
                         [-0.4]])
"""

def test():
    # test of the class PolyInfo
    print("=<><><><><><><><><><><>=")
    print("TESTING:  class PolyInfo")
    print("=<><><><><><><><><><><>=")
    info = PolyInfo()
    print("    Success in instantiation")
    #print("\nHere is the rootvec:")
    #print(info.all_bias)
    #print(len(info.all_bias))
    #print("Here is the index set")
    #print(info.idxset)
    #print(info.all_weight)
    #print(len(info.all_weight))

    # test one-dimensional case
    # degree 6 Legendre Polynomial 
    newidx = np.array([[6]])
    coefs = np.array([[0.25]])
    leg_info = PolyInfo(dim=1,polytype='leg',idxset=newidx,coefs=coefs,L=4)
    print("(+) Testing PolyInfo for Legendre")
    print("    The weight vec is")
    print(leg_info.all_weight)
    print("    The rootvec is")
    print(leg_info.all_bias)
    print("\n\n\n")

    # test TensorPolynomial
    print("==<><><><><><><><><><><><><><><><><>==")
    print("TESTING:        class TensorPolynomial")
    print("=<><><><><><><><><><><><><><><><><><>=")
    dim = 1
    bias_vec = leg_info.all_bias[0]
    weight_mat = leg_info.all_weight[0]
    num_L = 10
    net = TensorPolynomial(dim,bias_vec,weight_mat,num_L)
    print("    Successfully instantiated")
    print("(+) Testing poly_init()")
    net.poly_init()
    print("    Successfully performed initialization")
    print("(+) Print the min and max values of each subnet")
    for h in net.btp.hidden:
        max_val = h.b
        min_val = h.a
        print("    ",h._get_name())
        print("    in_N = ",h.in_N)
        print("    max = ",max_val)
        print("    min = ",min_val)

    # print the legnedre polynomial
    print("(+) Printing Legendre Polynomial")
    x = torch.linspace(-1,1,1001)
    x = torch.reshape(x,(1001,1))
    y = net(x)
    plt.plot(x.numpy(),y.detach().numpy())
    plt.show()

    # test PolyNet
    print("==<><><><><><><><><><><><><>==")
    print("TESTING:         class PolyNet")
    print("=<><><><><><><><><><><><><><>=")
    # set up the index set
    newidx = np.array([[0],[1],[2],[3],[4],[5],[6]])
    # find legendre interpolant of target function
    sx = np.linspace(-1,1,7)
    sy = 1./(1+20*sx**2)
    c = np.polynomial.legendre.legfit(sx,sy,6)
    coefs = c.reshape(7,1)
    leg_info = PolyInfo(dim=1,outdim=1,polytype='leg',idxset=newidx,coefs=coefs,L=3)
    net = PolyNet(leg_info)
    net.poly_init()

    print("(+) Printing the target and interpolant")
    # print the legendre interpolant
    nx = torch.linspace(-1,1,1001)
    nx = torch.reshape(nx,(1001,1))
    ny = net(nx)
    ty = 1./(1+20*nx**2)
    
    # plotting
    plt.plot(nx.numpy(),ny.detach().numpy())
    plt.plot(nx.numpy(),ty.numpy())
    plt.show()

if (__name__ == "__main__"):
    test()


