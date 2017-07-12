
## Your softmax(x) function should return a NumPy array of the same shape as x.
## For example, given a list or one-dimensional array (which is interpreted as a column vector representing a single sample), like:

import numpy as np
def rowsums(myarray):
    rsums = np.apply_along_axis( np.sum, axis=myarray.ndim-1, arr=myarray)
    print(rsums)
    print("rsums {}".format( rsums) )
    if rsums.ndim>0:
        rsums = rsums[:,np.newaxis]
    print(rsums)
    return rsums

def colsums(myarray):
    csums = np.apply_along_axis( np.sum, axis=0, arr=myarray)
    print(csums)
    print("rsums {}".format( csums) )
    print(csums)
    return csums


def softmax(pscores):
    pscores = np.array(pscores)
    print("scores.ndim {}".format( pscores.ndim) )
    escores = np.apply_along_axis( np.exp, axis=pscores.ndim-1, arr=pscores )
    print("escores {}".format( escores) )
    sums = colsums(myarray= escores)
    print("sums {}".format( sums) )
    prvals = escores/ sums
    return prvals

pscores = [1.0, 2.0, 3.0]
#pscores = np.array(pscores)
print( softmax(pscores ))

#Given a 2-dimensional array where each column represents a sample,
scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

res=  softmax(scores)
print(res )
