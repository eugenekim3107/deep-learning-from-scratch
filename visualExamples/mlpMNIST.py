import sys

sys.path.insert(0, "/Users/eugenekim/PycharmProjects/deepLearningFromScratch/mlp")

import numpy as np
import matplotlib.pyplot as plt
from Layer import Layer
from activationLayer import activationLayer
from NN import NN
from activations import ReLU, ReLU_prime, tanh, tanh_prime
from losses import mse, mse_prime
import torch
import os
import pandas as pd

os.chdir("..")

# prepare data
data = pd.read_csv("train.csv")
x_train = data.loc[:,"pixel0":].to_numpy().reshape(-1,1,784)
x_train = x_train.astype(np.float32)
x_train /= 255
temp = data.loc[:,"label"].to_numpy()
y_train = np.zeros((temp.shape[0], 10, 1))
for num in range(len(temp)):
    y_train[num][temp[num]] = 1
y_train = y_train.reshape(-1, 1, 10)

net = torch.load("mlp.obj")
out = net.predict(x_train)
print(out[0])
print(y_train[0])

