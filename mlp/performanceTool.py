import numpy as np

def accuracy(pred, true):
    pred = pred.reshape(-1, 10)
    true = true.reshape(-1, 10)
    full_pred = np.array([])
    full_true = np.array([])
    for i in range(len(pred)):
        max_val = pred[i][0]
        max_idx = 0
        true_val = 0
        for j in range(len(pred[i])):
            if true[i][j] == 1:
                true_val = j
            if pred[i][j] > max_val:
                max_val = pred[i][j]
                max_idx = j
        full_pred = np.append(full_pred, max_idx)
        full_true = np.append(full_true, true_val)
    return sum(full_pred == full_true) / len(full_true)
