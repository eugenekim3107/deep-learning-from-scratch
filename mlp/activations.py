import numpy as np

# define the activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_prime(x):
    return x > 0