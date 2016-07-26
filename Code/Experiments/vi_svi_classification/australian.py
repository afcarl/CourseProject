import numpy as np

from vi_svi_class_experiments import run_methods
from GP.covariance_functions import SquaredExponential, Matern, GammaExponential
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

model_params = np.array([1., 1., 1.])
model_covariance_obj = SquaredExponential(model_params)

maxiter = 30
max_out_iter = 4
max_inner_iter = 5
num_inputs = 20
batch_size = 20

x_tr, y_tr = load_svmlight_file('../../../../Programming/DataSets/Classification/australian_scale(690,14).txt')
x_tr, y_tr = shuffle(x_tr, y_tr, random_state=241)
data_name = 'australian'
file_name = data_name

x_tr = x_tr.toarray()
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr).T

x_tr = (x_tr + 1) / 2
y_tr = y_tr.reshape((y_tr.size, 1))
x_test = x_tr[:, int(x_tr.shape[1] * 0.8):]
y_test = y_tr[int(x_tr.shape[1] * 0.8):, :]
y_tr = y_tr[:int(x_tr.shape[1] * 0.8), :]
x_tr = x_tr[:, : int(x_tr.shape[1] * 0.8)]
dim, num = x_tr.shape
print(num, dim)


title = data_name + ' dataset, n = ' + str(num) + ', d = ' + str(dim)

opts = [{'mode': 'full', 'maxiter':maxiter, 'mydisp': True}, {'maxiter':max_inner_iter, 'mydisp': True},
        {'mode': 'adadelta', 'maxiter': 7,
         'verbose': True, 'batch_size': batch_size,
         'step_rate': 0.1}]

run_methods(x_tr, y_tr, x_test, y_test, model_params, opts, max_out_iter, file_name, num_inputs, title, True,
            adadelta=True, vi=True, svi=True)

