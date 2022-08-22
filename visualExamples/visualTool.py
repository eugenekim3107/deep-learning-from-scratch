import matplotlib.pyplot as plt
import numpy as np

def toolMNIST(x_train, y_train, y_pred):
    random = np.random.randint(0, x_train.shape[0])
    y_train =  y_train[random][0]
    y_pred = y_pred[random][0]
    max_val = y_pred[0]
    max_idx = 0
    true_val = y_train.argmax(axis=0)
    xreshaped = x_train[random].reshape(28,28)
    print("Predicted Value: " + str(max_idx) + "\n" + "True Value: " + str(true_val))
    plt.imshow(xreshaped, cmap="gray")
    plt.show()
    plt.close()
