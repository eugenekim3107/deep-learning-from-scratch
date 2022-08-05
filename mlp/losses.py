import numpy as np

# define the loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

def cross_entropy(y_true, y_pred):
    loss = -np.sum(y_true*np.log(y_pred, where=0))
    return loss/float(y_pred.shape[0])

def cross_entropy_prime(y_true, y_pred):
    (y_pred-y_true) / (1-y_pred)*y_pred