import matplotlib.pyplot as plt
import numpy as np

def toolMNIST(x_train, y_train, y_pred):
    fig = plt.figure()
    y_pred = np.argmax(np.array(y_pred), axis=2)
    for i in range(9):

        # get data
        random = np.random.randint(0, y_train.shape[0])
        y_train_temp = np.argmax(y_train[random])
        y_pred_temp = y_pred[random]
        xreshaped = x_train[random].reshape(28, 28)

        subplot = fig.add_subplot(3, 3, i+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.set_title(f"Pred:{int(y_pred_temp)}, Label:{y_train_temp}")
        subplot.imshow(xreshaped, cmap = plt.cm.gray_r)
    plt.savefig("visualExamples/samplePredictionsMNIST.png")
