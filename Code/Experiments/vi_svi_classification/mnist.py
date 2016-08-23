import numpy as np

from vi_svi_class_experiments import run_methods
from GP.covariance_functions import SquaredExponential, Matern, GammaExponential
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

model_params = np.array([10., 10., 0.01])
model_covariance_obj = SquaredExponential(model_params)

maxiter = 10
max_out_iter = 25
max_inner_iter = 10
num_inputs = 200
batch_size = 100

file_name = 'ijcnn'

x_tr, y_tr = load_svmlight_file('../../../../Programming/DataSets/Classification/multiclass/mnist_scale(60000,780).txt')
x_tr, y_tr = shuffle(x_tr, y_tr, random_state=241)
x_test, y_test = load_svmlight_file('../../../../Programming/DataSets/Classification/multiclass/mnist_scale_test(10000,780).txt')
data_name = 'mnist'

x_tr = x_tr.toarray()
x_test = x_test.toarray()
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr).T
x_test = scaler.transform(x_test).T

y_tr = (y_tr %2 == 0).astype(float)
y_test = (y_test %2 == 0).astype(float)

x_tr = (x_tr + 1) / 2
x_test = (x_test + 1) / 2
y_tr = y_tr[:, None]
y_test = y_test[:, None]
y_tr[y_tr == 0] = -1
y_test[y_test == 0] = -1
dim, num = x_tr.shape
print(num, dim)


title = data_name + ' dataset, n = ' + str(num) + ', d = ' + str(dim)

opts = [{'mode': 'full', 'maxiter':maxiter, 'mydisp': True}, {'maxiter':max_inner_iter, 'mydisp': True},
        {'mode': 'adadelta', 'maxiter': 5,
         'verbose': True, 'batch_size': batch_size,
         'step_rate': 0.05}]

run_methods(x_tr, y_tr, x_test, y_test, model_params, opts, max_out_iter, file_name, num_inputs, title, True,
            adadelta=True, vi=False, svi=False)

