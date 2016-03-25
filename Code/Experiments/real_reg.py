import time

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from GP.covariance_functions import SquaredExponential
from GP.gaussian_process_regression import GPR

#Parameters
# random_seed_w0 = 32
mu, sigma1 = 0, 10

# x_tr, y_tr = load_svmlight_file('../../Programming/DataSets/Regression/abalone(4177, 8).txt')
# data_name = 'abalone'
# x_tr, y_tr = load_svmlight_file('../../Programming/DataSets/Regression/cpusmall(8192, 12).txt')
# data_name = 'cpusmall'
x_tr, y_tr = load_svmlight_file('../../Programming/DataSets/Regression/bodyfat(252, 14).txt')
data_name = 'bodyfat'


x_tr = x_tr.T
x_tr = x_tr.toarray()
scaler = StandardScaler()
x_tr = scaler.fit_transform(x_tr)
# y_g = y_g.toarray()


x_tr = (x_tr + 1) / 2
y_tr = y_tr.reshape((y_tr.size, 1))
x_test = x_tr[:, int(x_tr.shape[1] * 0.8):]
y_test = y_tr[int(x_tr.shape[1] * 0.8):, :]
y_tr = y_tr[:int(x_tr.shape[1] * 0.8), :]
x_tr = x_tr[:, : int(x_tr.shape[1] * 0.8)]

print('Data set', data_name)
print("Number of data points: ", x_tr.shape[1])
print("Number of test points: ", x_test.shape[1])
print("Number of features: ", x_tr.shape[0])
print()
# #Generating the starting point
# np.random.seed(random_seed_w0)
# w0 = np.random.rand(3)

model_params = np.array([1.,  0.5,  1.])
model_covariance_obj = SquaredExponential(model_params)

model_params = np.array([0.6, 0.3, 0.1])
model_covariance_obj = SquaredExponential(model_params)
num = 200
test_num = 100
dim = 1
seed = 21
method = 'means'  # possible methods: 'brute', 'vi', 'means', 'svi'
parametrization = 'natural'  # possible parametrizations for svi method: cholesky, natural
ind_inputs_num = 30
max_iter = 100

if method == 'brute':
    new_gp = GPR(model_covariance_obj)
    new_gp.fit(x_tr, y_tr, max_iter=max_iter)
    predicted_y_test, high, low = new_gp.predict(x_test, x_tr, y_tr)

elif method == 'means' or method == 'vi':
    model_covariance_obj = SquaredExponential(model_params)
    new_gp = GPR(model_covariance_obj, method=method)
    start = time.time()
    new_gp.fit(x_tr, y_tr, num_inputs=ind_inputs_num, max_iter=max_iter)
    print(time.time() - start)
    inducing_points, mean, cov = new_gp.inducing_inputs
    predicted_y_test, high, low = new_gp.predict(x_test)

elif method == 'svi':
    model_covariance_obj = SquaredExponential(model_params)
    new_gp = GPR(model_covariance_obj, method=method, parametrization=parametrization)
    new_gp.fit(x_tr, y_tr, num_inputs=ind_inputs_num, max_iter=max_iter)
    inducing_points, mean, cov = new_gp.inducing_inputs
    predicted_y_test, high, low = new_gp.predict(x_test)


print('RMSE:', np.linalg.norm(y_test - predicted_y_test) / np.sqrt(y_test.size))
print('r2_score', r2_score(y_test, predicted_y_test))