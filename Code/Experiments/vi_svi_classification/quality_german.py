import numpy as np

from quality_vi_svi_class_experiments import run_methods
from GP.covariance_functions import SquaredExponential, Matern, GammaExponential
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler

model_params = np.array([1., 1.5, 0.3])
model_covariance_obj = SquaredExponential(model_params)

maxiter = 100
input_nums = [10, 25, 50, 100, 200]

file_name = 'class_german'

opts = {'maxiter':maxiter, 'mydisp': True}

x_tr, y_tr = load_svmlight_file('../../../../Programming/DataSets/Classification/german_numer.txt')
data_name = 'German numer'

x_tr = x_tr.T
x_tr = x_tr.toarray()
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)

x_tr = (x_tr + 1) / 2
y_tr = y_tr.reshape((y_tr.size, 1))
x_test = x_tr[:, int(x_tr.shape[1] * 0.8):]
y_test = y_tr[int(x_tr.shape[1] * 0.8):, :]
y_tr = y_tr[:int(x_tr.shape[1] * 0.8), :]
x_tr = x_tr[:, : int(x_tr.shape[1] * 0.8)]
dim, num = x_tr.shape

title = 'german numer dataset, n = ' + str(num) + ', d = ' + str(dim)

run_methods(x_tr, y_tr, x_test, y_test, model_params, input_nums, file_name, title, True)

