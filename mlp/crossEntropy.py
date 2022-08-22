import numpy as np

def cross_entropy(X, y):
    softmax = np.exp(X - np.max(X)) / np.sum(np.exp(X - np.max(X)), axis=1)
    return np.sum(np.multiply(-np.log(softmax), y))

def cross_entropy_prime(X, y):
    softmax = np.exp(X - np.max(X)) / np.sum(np.exp(X - np.max(X)), axis=1)
    grad = softmax - y
    return grad
