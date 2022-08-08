from Layer import Layer
from activationLayer import activationLayer
from NN import NN
from activations import ReLU, ReLU_prime, tanh, tanh_prime
from losses import mse, mse_prime
import numpy as np
import pandas as pd
import os
import torch

# data file
os.chdir("..")
data = pd.read_csv("test.csv")
x_test = data.loc[:,"pixel0":].to_numpy().reshape(-1,1,784)
x_test = x_test.astype(np.float32)
x_test /= 255
temp = data.loc[:,"label"].to_numpy()
y_test = np.zeros((temp.shape[0], 10, 1))
for num in range(len(temp)):
    y_test[num][temp[num]] = 1
y_test = y_test.reshape(-1,1,10)

# nn model
net = torch.load("mlpMNIST.obj")
pred = net.predict(x_test)
