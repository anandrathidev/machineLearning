import numpy as np

def softmax(pscores):
    pscores = np.array(pscores)
    escores = np.apply_along_axis( np.exp, axis=scores.ndim-1, arr=pscores )
    print(escores)
    rsums = np.apply_along_axis( np.sum, axis=escores.ndim-1, arr=escores )
    print(rsums)
    prvals = escores/ rsums
    return prvals    

pscores = [1.0, 2.0, 3.0]
pscores = np.array(pscores)
print( softmax(pscores ))

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

print( softmax(scores))
