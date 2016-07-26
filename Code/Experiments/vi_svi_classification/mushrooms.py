import numpy as np

from vi_svi_class_experiments import run_methods
from GP.covariance_functions import SquaredExponential, Matern, GammaExponential
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

model_params = np.array([2., 2., 1.])
model_covariance_obj = SquaredExponential(model_params)

maxiter = 5
max_out_iter = 4
max_inner_iter = 3
num_inputs = 35
batch_size = 50

file_name = 'class_mushrooms'

opts = [{'mode': 'full', 'maxiter':maxiter, 'mydisp': True}, {'maxiter':max_inner_iter, 'mydisp': True},
        {'mode': 'adadelta', 'maxiter': 7,
         'verbose': True, 'batch_size': batch_size,
         'step_rate': 0.3}]

x_tr, y_tr = load_svmlight_file('../../../../Programming/DataSets/Classification/mushrooms(8124,112).txt')
x_tr, y_tr = shuffle(x_tr, y_tr, random_state=241)
y_tr[y_tr == 2] = -1
print(y_tr[-5:])
# exit(0)
data_name = 'mushrooms'

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
print(num, dim)

title = data_name + ' dataset, n = ' + str(num) + ', d = ' + str(dim)

run_methods(x_tr, y_tr, x_test, y_test, model_params, opts, max_out_iter, file_name, num_inputs, title, True,
            adadelta=True, vi=True, svi=True)

