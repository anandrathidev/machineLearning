# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 15:30:04 2017

@author: anandrathi
"""
import numpy as np


"""
Q1 
from i = 0 to 12  3000(1 + i/12 ) ^i 
"""
vec1 = np.array(range(1,13))

def IntrestRateCalc(i):
    """"""
    return pow(3000 * (1 + i/12 ) , i)

vec1_i  = np.apply_along_axis(IntrestRateCalc, 0, vec1)


"""
Q2 
from i = 1 to 35  2^1/i + 3^1/(i^2)
"""

vec2 = np.array(range(1,26))

def SecondQFunc(i):
    """"""
    return ( pow(2 , 1/i) + pow(2 , 1/pow(i,3)))

vec2_i  = np.apply_along_axis(SecondQFunc, 0, vec2)


"""
Q2 

6X10 Matrix of random numbers between 1 - 10

"""


vec3 =  np.random.randint(1,11, size=(6, 10))
def rowGT5(i):
    """"""
    r= 1 if i>5  else 0 
    return (r)

def rowGT5Test(i):
    """"""
    r= 1 if i>5  else 0 
    return (r)


vec3_i  = np.apply_along_axis(rowGT5, 1, vec3)

list(map( rowGT5, list(vec3) ))
