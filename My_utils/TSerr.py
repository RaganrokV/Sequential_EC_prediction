#%%
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score



def dist_err(y_pred, y_real, method):
    if method == "ME":
        D_err = np.mean(y_pred - y_real)

    elif method == "MAE":
        D_err = mean_absolute_error(y_pred, y_real)

    elif method == "MSE":
        D_err = mean_squared_error(y_pred, y_real)

    elif method == "RMSE":
        D_err = sqrt(mean_absolute_error(y_pred, y_real))

    elif method == "MAPE":
        D_err = np.mean(np.abs((y_pred - y_real) / y_real)) * 100

    elif method == "MLE":
        D_err = np.mean(np.log(y_pred) - np.log(y_real))

    elif method == "MALE":
        D_err = mean_absolute_error(np.log(y_pred), np.log(y_real))

    elif method == "MSLE":
        D_err = mean_squared_error(np.log(y_pred), np.log(y_real))

    elif method == "RMSLE":
        D_err = sqrt(mean_absolute_error(np.log(y_pred), np.log(y_real)))

    elif method == "MdE":
        D_err = np.median(y_pred - y_real)

    elif method == "MAdE":
        D_err = np.median(np.abs(y_pred - y_real))

    elif method == "MSdE":
        D_err = np.median(sqrt(y_pred - y_real))

    elif method == "R2":
        D_err = r2_score(y_pred, y_real)

    elif method == "r":
        D_err = sqrt(r2_score(y_pred, y_real))

    elif method == "ED":
        D_err = sqrt(np.sum((y_pred - y_real) ** 2))

    elif method == "PD":
        # mapping y_pred
        temp_pred = y_pred[1:] - y_pred[:-1]
        temp_up = np.where(temp_pred <= 0, temp_pred, 1, )
        p_pred = np.where(temp_up >= 0, temp_up, -1)
        # mapping y_real
        temp_real = y_real[1:] - y_real[:-1]
        temp_up = np.where(temp_real <= 0, temp_real, 1, )
        p_real = np.where(temp_up >= 0, temp_up, -1)
        # equal weights
        D_err = mean_absolute_error(p_pred, p_real)

    elif method == "SD":
        # mapping y_pred
        temp_pred = y_pred[1:] - y_pred[:-1]
        temp_up = np.where(temp_pred <= 0, temp_pred, 1, )
        p_pred = np.where(temp_up >= 0, temp_up, -1)
        # mapping y_real
        temp_real = y_real[1:] - y_real[:-1]
        temp_up = np.where(temp_real <= 0, temp_real, 1, )
        p_real = np.where(temp_up >= 0, temp_up, -1)
        # equal weights
        amp_err = np.abs(temp_pred - temp_real)
        dir_err = np.abs(p_pred - p_real)
        D_err = (np.sum(amp_err * dir_err)) / len(dir_err)

    return D_err