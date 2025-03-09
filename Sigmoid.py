import numpy as np

def sigmoid(x):
    """
    Implementation of the sigmoid function
    x is a numpy array
    """
    return np.exp(-np.logaddexp(0, -x))   ## using this to avoid overflow