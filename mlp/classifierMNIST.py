from Layer import Layer
from activationLayer import activationLayer
from NN import NN
from activations import ReLU, ReLU_prime
from losses import cross_entropy, cross_entropy_prime
import numpy as np
import pandas as pd
import os

# data file
os.chdir("..")
data = pd.read_csv("train.csv")
print(data)
# training data
# x_train = pd.read_csv("train.csv")
# print(x_train)