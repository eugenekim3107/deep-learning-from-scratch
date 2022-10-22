import numpy as np

# define the loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def cross_entropy(y_true, y_pred):
    softmax = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1)
    return -np.sum(np.log(softmax) * y_true)

def cross_entropy_prime(y_true, y_pred):
    softmax = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1)
    grad = softmax - y_true
    return grad