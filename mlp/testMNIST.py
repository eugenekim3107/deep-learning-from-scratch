from Layer import Layer
from activationLayer import activationLayer
from NN import NN
from activations import ReLU, ReLU_prime, tanh, tanh_prime
from losses import mse, mse_prime
from performanceTool import accuracy
import numpy as np
import pandas as pd
import os
import torch

# data file
os.chdir("..")
data = pd.read_csv("train.csv")
x_train = data.loc[:,"pixel0":].to_numpy().reshape(-1,1,784)
x_train = x_train.astype(np.float32)
x_train /= 255
temp = data.loc[:,"label"].to_numpy()
y_train = np.zeros((temp.shape[0], 10, 1))
for num in range(len(temp)):
    y_train[num][temp[num]] = 1
y_train = y_train.reshape(-1,1,10)

# nn model
net = torch.load("mlp/mlpMNIST.obj")

# predictions
pred = np.array(net.predict(x_train))
print(accuracy(pred, y_train))
