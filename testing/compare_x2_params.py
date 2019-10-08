#!/usr/bin/python3
"""
file: compare_x2_params.py

Here we compare the parameterizations of x^2 either by scaling
the input or by using a network which can approximatoin x^2 on [a,b]
"""
import sys
sys.path.append("../")

from polyinit import *

net = PolyNet()
