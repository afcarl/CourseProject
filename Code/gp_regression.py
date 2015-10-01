import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from sklearn import svm
import time

from covariance_functions import delta, covariance_mat
# from gp_reg_data import x_g, y_g, plot_data, x_test, y_test
from reg_parameters import model_params, common_params
# from optimization import full_gradient_descent, Problem
from real_reg_data import x_g, y_g, x_test, y_test


def sample_for_matrices(mean_vec, cov_mat):
    y = np.random.multivariate_normal(mean_vec.reshape((mean_vec.size,)), cov_mat)
    # upper_bound = mean_vec + 3 * np.sqrt(np.diagonal(cov_mat).reshape(mean_vec.shape))
    # lower_bound = mean_vec - 3 * np.sqrt(np.diagonal(cov_mat).reshape(mean_vec.shape))
    return y

def mean_square_loss(true_y, app_y):
    return np.linalg.norm(true_y - app_y)

#Reassigning the parameters
# density = common_params.density
d, n = x_g.shape
m = np.vectorize(lambda x: 0)
covariance_obj = model_params.cov_obj
K = covariance_obj.covariance_function
ml = covariance_obj.oracle

clf = svm.SVC()
clf.fit(x_g.T, y_g.reshape((y_g.size,)))
svm_y_test = clf.predict(x_test.T)

#Fitting
K_x = covariance_mat(K, x_g, x_g)
I = np.eye(K_x.shape[0])
anc_mat = np.linalg.inv(K_x)


def fun(w):
    loss, grad = ml(x_g, y_g, w)
    return loss


def oracle_fun(w):
    return ml(x_g, y_g, w)


def grad(w):
    loss, grad = ml(x_g, y_g, w)
    return grad


# prb = Problem(oracle_fun, covariance_obj.get_params())
# point_v, time_v, loss_v = full_gradient_descent(prb, grad_eps=1e-2, max_iter=200,  max_time=np.inf, freq=50)
# optimal_params = point_v[len(point_v) - 1]
# print(optimal_params)

bnds = ((1e-2, None), (1e-2, None), (1e-2, None))
start = time.clock()
res = op.minimize(fun, covariance_obj.get_params(), args=(), method='L-BFGS-B', jac=grad, bounds=bnds,
            options={'ftol':1e-3, 'disp':False})
optimal_params = res.x
timing = time.clock() - start
print(timing)

covariance_obj.set_params(optimal_params)
K = covariance_obj.covariance_function

K_x = covariance_mat(K, x_g, x_g)
I = np.eye(K_x.shape[0])
anc_mat = np.linalg.inv(K_x)

K_x_test = covariance_mat(K, x_g, x_test)
K_test_x = covariance_mat(K, x_test, x_g)
K_test = covariance_mat(K, x_test, x_test)

new_mean = np.dot(np.dot(K_test_x, anc_mat), y_g)
new_cov = K_test - np.dot(np.dot(K_test_x, anc_mat), K_x_test)
app_y_test = sample_for_matrices(new_mean, new_cov)

print("SVM: ", np.linalg.norm(y_test - svm_y_test))
print("GP: ", np.linalg.norm(y_test - app_y_test))
