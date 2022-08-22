from Layer import Layer
from activationLayer import activationLayer
from NN import NN
from activations import ReLU, ReLU_prime, tanh, tanh_prime
from losses import mse, mse_prime, cross_entropy, cross_entropy_prime
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

# network
net = NN()
net.add(Layer(784, 100))
net.add(activationLayer(ReLU, ReLU_prime))
net.add(Layer(100,10))
net.add(activationLayer(tanh, tanh_prime))

# train
net.use(cross_entropy, cross_entropy_prime)
net.fit(x_train[0:1000], y_train[0:1000], epochs=50, learning_rate=0.25)

# test
out = net.predict(x_train)
print(out[0])
print(y_train[0])

# save neural network
torch.save(net, "mlp/mlpMNIST.obj")
