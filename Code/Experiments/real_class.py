import numpy as np
from old_version_gp import GaussianProcess
from sklearn.datasets import load_svmlight_file

from GP.covariance_functions import SquaredExponential

#Parameters
# random_seed_w0 = 32
mu, sigma1 = 0, 10

x_tr, y_tr = load_svmlight_file('Data/Classification/australian.txt')
x_tr = x_tr.T
x_tr = x_tr.toarray()
# y_g = y_g.toarray()
data_name = 'Australian'

x_tr = (x_tr + 1) / 2
y_tr = y_tr.reshape((y_tr.size, 1))
x_test = x_tr[:, int(x_tr.shape[1] * 0.8):]
y_test = y_tr[int(x_tr.shape[1] * 0.8):, :]
y_tr = y_tr[:int(x_tr.shape[1] * 0.8), :]
x_tr = x_tr[:, : int(x_tr.shape[1] * 0.8)]

print("Number of data points: ", x_tr.shape[1])
print("Number of test points: ", x_test.shape[1])
print("Number of features: ", x_tr.shape[0])
# #Generating the starting point
# np.random.seed(random_seed_w0)
# w0 = np.random.rand(3)

model_params = np.array([10.,  0.7,  3.])
model_covariance_obj = SquaredExponential(model_params)
new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'class')
new_gp.find_hyper_parameters(x_tr, y_tr, max_iter=100)
print(new_gp.covariance_obj.get_params())
predicted_y_test = new_gp.predict(x_test, x_tr, y_tr)

print("Mistakes: ", np.sum(predicted_y_test != y_test))
