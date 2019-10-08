#!/usr/bin/python3
"""
roots.py

Tools to generate lists of roots of several polynomials.
Lists of roots are stored in the data directory
"""

import numpy as np
import numpy.polynomial.legendre as leg
import numpy.polynomial.chebyshev as cheb

# set some global variables
MAXD = 20 # maximum degree allowed

# function to genereate Legendre 
def leg_roots_vec():
    roots = {}
    # loop over all degrees up to MAXD
    for i in range(0,MAXD):
        c = np.zeros(MAXD)
        c[i] = 1
        r = leg.legroots(c)
        roots.update({i:r})
   
    return roots

# function to get scaling factor in legendre polynomial
def get_leg_scalfac(deg):
    testpt = 0.5
    c = np.zeros(MAXD)
    c[deg] = 1
    val = leg.legval(testpt,c,tensor=False)
    r = leg.legroots(c)
    prod = 1
    for root in r:
        prod = prod * (testpt-root)

    scalfac = val/prod

    return scalfac

# function to genereate Chebyshev Roots 
def cheb_roots_vec():
    roots = {}
    # loop over all degrees up to MAXD
    for i in range(0,MAXD):
        c = np.zeros(MAXD)
        c[i] = 1
        r = cheb.chebroots(c)
        roots.update({i:r})
   
    return roots 

def test():
    # legendre roots
    try:
        lr = leg_roots_vec()
        print("PASSED: leg_rotos_vec()")
    except:
        print("FAILED: leg_roots_vec()")
    # scaling factor legendre
    try:
        s = get_leg_scalfac(3)
        print("PASSED: get_leg_scalfac()")
    except:
        print("FAILED: get_leg_scalfac()")
    # cheb roots
    try:
        lr = cheb_roots_vec()
        print("PASSED: cheb_rotos_vec()")
    except:
        print("FAILED: cheb_roots_vec()")

if (__name__ == '__main__'):
    test()
