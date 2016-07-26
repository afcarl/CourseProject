import numpy as np

from vi_svi_class_experiments import run_methods
from GP.covariance_functions import SquaredExponential, Matern, GammaExponential
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

model_params = np.array([1., 1., 1.])
model_covariance_obj = SquaredExponential(model_params)

maxiter = 30
max_out_iter = 8
max_inner_iter = 3
num_inputs = 50
batch_size = 100

file_name = 'class_mushrooms'

# x_tr, y_tr = load_svmlight_file('../../../../Programming/DataSets/Classification/magic_gamma_telescope(19020,11).txt')
data = np.load('../../../../Programming/DataSets/Classification/magic_gamma_telescope(19020,11).npy')
x_tr, y_tr = data[:, :-1], data[:, -1]
x_tr, y_tr = shuffle(x_tr, y_tr, random_state=241)
data_name = 'magic telescope'

# x_tr = x_tr.T
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr).T
print(x_tr.shape)
print(y_tr.shape)

x_tr = (x_tr + 1) / 2
y_tr = y_tr[:, None]
y_tr[y_tr == 0] = -1
x_test = x_tr[:, int(x_tr.shape[1] * 0.8):]
y_test = y_tr[int(x_tr.shape[1] * 0.8):, :]
y_tr = y_tr[:int(x_tr.shape[1] * 0.8), :]
x_tr = x_tr[:, : int(x_tr.shape[1] * 0.8)]
dim, num = x_tr.shape
print(num, dim)

title = data_name + ' dataset, n = ' + str(num) + ', d = ' + str(dim)

opts = [{'mode': 'full', 'maxiter':maxiter, 'mydisp': True}, {'maxiter':max_inner_iter, 'mydisp': True},
        {'mode': 'adadelta', 'maxiter': 22,
         'verbose': True, 'batch_size': batch_size,
         'step_rate': 0.1}]

run_methods(x_tr, y_tr, x_test, y_test, model_params, opts, max_out_iter, file_name, num_inputs, title, True,
            adadelta=True, svi=True, vi=True)

