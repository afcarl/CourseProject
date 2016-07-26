import numpy as np

from vi_svi_class_experiments import run_methods
from GP.covariance_functions import SquaredExponential, Matern, GammaExponential
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

model_params = np.array([2., 2., 1.])
model_covariance_obj = SquaredExponential(model_params)

maxiter = 12
max_out_iter = 10
max_inner_iter = 2
num_inputs = 25
batch_size = 50

file_name = 'class_mushrooms'

opts = [{'maxiter':maxiter, 'mydisp': True}, {'maxiter':max_inner_iter, 'mydisp': True}]

x_tr, y_tr = load_svmlight_file('../../../../Programming/DataSets/Classification/svmguide1(3089,4).txt')
x_tr, y_tr = shuffle(x_tr, y_tr, random_state=241)
x_test, y_test = load_svmlight_file('../../../../Programming/DataSets/Classification/svmguide1_test(4000,4).txt')
data_name = 'svmguide'
print(x_test.shape)

# x_tr = x_tr.T
x_tr = x_tr.toarray()
# x_test = x_test.T
x_test = x_test.toarray()
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr).T
x_test = scaler.transform(x_test).T

x_tr = (x_tr + 1) / 2
x_test = (x_test + 1) / 2
y_tr = y_tr[:, None]
y_test = y_test[:, None]
y_tr[y_tr == 0] = -1
y_test[y_test == 0] = -1
# print(y_tr[-10:])
# exit(0)
# x_test = x_tr[:, int(x_tr.shape[1] * 0.8):]
# y_test = y_tr[int(x_tr.shape[1] * 0.8):, :]
# y_tr = y_tr[:int(x_tr.shape[1] * 0.8), :]
# x_tr = x_tr[:, : int(x_tr.shape[1] * 0.8)]
dim, num = x_tr.shape
print(num, dim)


title = data_name + ' dataset, n = ' + str(num) + ', d = ' + str(dim)

opts = [{'mode': 'full', 'maxiter':maxiter, 'mydisp': True}, {'maxiter':max_inner_iter, 'mydisp': True},
        {'mode': 'adadelta', 'maxiter': 5,
         'verbose': True, 'batch_size': batch_size,
         'step_rate': 0.1}]

run_methods(x_tr, y_tr, x_test, y_test, model_params, opts, max_out_iter, file_name, num_inputs, title, True,
            adadelta=True, vi=True, svi=True)

