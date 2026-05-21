import numpy as np
import torch
import torch.nn.functional as F


def LSD_l_mag_db (y_true, y_pred):
    """ for input data is log-magnitude"""
    y_true = y_true.reshape(-1, 1)
    y_pred = y_pred.reshape(-1, 1)
    y_true = y_true[:,0]
    y_pred = y_pred[:,0]
    error = y_pred - y_true
    Lsd = np.pow(error,2)
    Lsd = np.sqrt(np.sum(Lsd)/np.shape(Lsd)[0])

    return Lsd

def calculate_rmse(predictions, targets):
    mse = F.mse_loss(predictions, targets, reduction='mean') # Calculate the mean squared error
    rmse = torch.sqrt(mse) # Take the square root
    return rmse