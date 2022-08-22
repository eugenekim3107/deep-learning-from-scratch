import numpy as np

# define the loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def cross_entropy(X, y):
    softmax = np.exp(X - np.max(X)) / np.sum(np.exp(X - np.max(X)), axis=1)
    return np.sum(np.multiply(-np.log(softmax), y))

def cross_entropy_prime(X, y):
    softmax = np.exp(X - np.max(X)) / np.sum(np.exp(X - np.max(X)), axis=1)
    grad = softmax - y
    return grad