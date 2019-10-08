#!/usr/bin/python3
"""
file: defaults.py

useful arrays for setting the default values of some classes
"""
import numpy as np

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


