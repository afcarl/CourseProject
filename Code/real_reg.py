from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn import svm, cross_validation

from gaussian_process import GaussianProcess, gp_plot_reg_data, gp_plot_class_data
from covariance_functions import CovarianceFamily, SquaredExponential

#Parameters
# random_seed_w0 = 32
mu, sigma1 = 0, 10

x_tr, y_tr = load_svmlight_file('Data/housing.txt')
x_tr = x_tr.T
x_tr = x_tr.toarray()
# y_g = y_g.toarray()
data_name = 'Housing'

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

model_params = np.array([1.,  0.5,  1.])
model_covariance_obj = SquaredExponential(model_params)
new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
new_gp.find_hyper_parameters(x_tr, y_tr)
print(new_gp.covariance_obj.get_params())
predicted_y_test = new_gp.predict(x_test, x_tr, y_tr)

n_samples = x_tr.shape[1]
max_score = -np.inf
c_max, eps_max = 0, 0
for c, eps in list(zip([1, 10, 1000, 10000], [10, 1, 0.5, 0.1, 0.01])):
    clf = svm.SVR(C=c, epsilon=eps)
    cv = cross_validation.ShuffleSplit(n_samples, n_iter=3, test_size=0.3, random_state=0)
    scores = cross_validation.cross_val_score(clf, x_tr.T, y_tr.reshape((y_tr.size,)), cv=5)
    if scores.mean() > max_score:
        max_score = scores.mean()
        c_max, eps_max = c, eps
print(c_max, eps_max)
clf = svm.SVR(C=c_max, epsilon=eps_max)
clf.fit(x_tr.T, y_tr.reshape((y_tr.size,)))
svm_y_test = clf.predict(x_test.T)

print(np.linalg.norm(y_test - svm_y_test) / y_test.size)
print(np.linalg.norm(y_test - predicted_y_test) / y_test.size)