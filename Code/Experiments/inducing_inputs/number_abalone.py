import numpy as np

from inducing_inputs_experiments import run_methods
from GP.gaussian_process_regression import GPR
from GP.covariance_functions import SquaredExponential, Matern, GammaExponential
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler

model_params = np.array([2.0, 0.5, 0.5])
model_covariance_obj = SquaredExponential(model_params)
file_name = 'abalone'
input_nums = [10, 20, 50, 100, 200, 300, 400, 500, 600]
# input_nums = [300, 400, 500]

x_tr, y_tr = load_svmlight_file('../../../../Programming/DataSets/Regression/abalone(4177, 8).txt')
data_name = 'abalone'
title='abalone dataset, n = 3342, d = 8'

x_tr = x_tr.T
x_tr = x_tr.toarray()
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)
y_tr = scaler.fit_transform(y_tr)

x_tr = (x_tr + 1) / 2
y_tr = y_tr.reshape((y_tr.size, 1))
x_test = x_tr[:, int(x_tr.shape[1] * 0.8):]
y_test = y_tr[int(x_tr.shape[1] * 0.8):, :]
y_tr = y_tr[:int(x_tr.shape[1] * 0.8), :]
x_tr = x_tr[:, : int(x_tr.shape[1] * 0.8)]

run_methods(x_tr, y_tr, x_test, y_test, model_params, input_nums, file_name, title, True, full=False, vi=False)
