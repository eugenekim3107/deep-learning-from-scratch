import numpy as np

def cross_entropy(x):
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)), axis=1)

def softmax_prime(x):
    return
