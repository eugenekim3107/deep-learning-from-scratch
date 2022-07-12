import numpy as np
import pandas as pd


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def ReLU(z):
    return np.maximum(0, z)


def ReLU_deriv(z):
    return z > 0


class NeuralNetwork:
    def __init__(self, x, y, hidden_size, alpha):
        self.w1 = np.random.rand(hidden_size, x.shape[0]) - 0.5
        self.b1 = np.random.rand(hidden_size, 1) - 0.5
        self.w2 = np.random.rand(hidden_size//2, hidden_size) - 0.5
        self.b2 = np.random.rand(hidden_size//2, 1) - 0.5
        self.w3 = np.random.rand(10, hidden_size//2) - 0.5
        self.b3 = np.random.rand(10, 1) - 0.5
        self.alpha = alpha
        self.m = x.shape[0]
        self.x = x
        self.y = y
        self.z1 = 0
        self.a1 = 0
        self.z2 = 0
        self.a2 = 0
        self.z3 = 0
        self.a3 = 0
        self.dz3 = 0
        self.dw3 = 0
        self.db3 = 0
        self.dz2 = 0
        self.dw2 = 0
        self.db2 = 0
        self.dz1 = 0
        self.dw1 = 0
        self.db1 = 0

    def forward(self):
        self.z1 = self.w1.dot(self.x) + self.b1
        self.a1 = ReLU(self.z1)
        self.z2 = self.w2.dot(self.a1) + self.b2
        self.a2 = ReLU(self.z2)
        self.z3 = self.w3.dot(self.a2) + self.b3
        self.a3 = softmax(self.z3)
        return self.a3

    def backprop(self):
        self.dz3 = self.a3 - self.y
        self.dw3 = (1 / self.m) * self.dz3.dot(self.a2.T)
        self.db3 = (1 / self.m) * np.sum(self.dz3)
        self.dz2 = self.dw3.T.dot(self.dz3) * ReLU_deriv(self.z2)
        self.dw2 = (1 / self.m) * self.dz2.dot(self.a1.T)
        self.db2 = (1 / self.m) * np.sum(self.dz2)
        self.dz1 = self.dw2.T.dot(self.dz2) * ReLU_deriv(self.z1)
        self.dw1 = (1 / self.m) * self.dz1.dot(self.x.T)
        self.db1 = (1 / self.m) * np.sum(self.dz1)

    def reset_grad(self):
        self.dz3 = 0
        self.dw3 = 0
        self.db3 = 0
        self.dz2 = 0
        self.dw2 = 0
        self.db2 = 0
        self.dz1 = 0
        self.dw1 = 0
        self.db1 = 0

    def update(self):
        self.w1 = self.w1 - self.alpha * self.dw1
        self.b1 = self.b1 - self.alpha * self.db1
        self.w2 = self.w2 - self.alpha * self.dw2
        self.b2 = self.b2 - self.alpha * self.db2
        self.w3 = self.w3 - self.alpha * self.dw3
        self.b3 = self.b3 - self.alpha * self.db3
