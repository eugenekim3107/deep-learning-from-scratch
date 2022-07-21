from Layer import Layer
from activationLayer import activationLayer
from regressionNN import regressionNN
from activations import tanh, tanh_prime
from losses import mse, mse_prime
import numpy as np

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = regressionNN()
net.add(Layer(2, 3))
net.add(activationLayer(tanh, tanh_prime))
net.add(Layer(3, 1))
net.add(activationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)
