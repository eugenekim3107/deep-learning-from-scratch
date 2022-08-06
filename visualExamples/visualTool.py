import matplotlib.pyplot as plt
import numpy as np

def toolMNIST(x_train, y_train, y_pred):
    random = np.random.randint(0, x_train.shape[0])
    y_train =  y_train[random][0]
    y_pred = y_pred[random][0]
    max_val = y_pred[0]
    max_idx = 0
    true_val = 0
    xreshaped = x_train[random].reshape(28,28)
    for i in range(len(y_pred)):
        if y_train[i] == 1:
            true_val = i
        if y_pred[i] > max_val:
            max_val = y_pred[i]
            max_idx = i
    print("Predicted Value: " + str(max_idx) + "\n" + "True Value: " + str(true_val))
    plt.imshow(xreshaped, cmap="gray")
    plt.show()
