import random
import math
import numpy as np

arr1 = np.array([[0, 1, 2],
                 [2, 1, 0],
                 [0, 2, 0]])
arr2 = np.array([[0, 0, 1],
                 [0, 0, 0]])
b = np.array([1, 2, 3, 4])
def gd_softmax(y_pdt):
    sm = y_pdt.reshape(-1, 1)
    return np.diagflat(sm) - sm.dot(sm.T)
print(gd_softmax(b))

