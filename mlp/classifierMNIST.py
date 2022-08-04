from Layer import Layer
from activationLayer import activationLayer
from classifierNN import classifierNN
from activations import ReLU, ReLU_prime
from losses import cross_entropy, cross_entropy_prime
import numpy as np
import pandas as pd
import os

# data file
print(os.getcwd())
# training data
# x_train = pd.read_csv("train.csv")
# print(x_train)