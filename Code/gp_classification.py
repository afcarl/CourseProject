import numpy as np
import matplotlib.pyplot as plt
from covariance_functions import covariance_mat
from gp_class_data import x_g, y_g, plot_data, x_test, y_test, sigmoid
from class_parameters import model_params, common_params
import scipy.optimize as opt
from scipy.optimize import check_grad


def logistic_loss(f, y):
    """log p(y|f)"""
    f = f.reshape((f.size, 1))
    return np.sum(np.log(np.exp(-y * f) + np.ones(y.shape)))


def logistic_likelyhood_hessian(f):
    """Hessian of the log p(y|f)"""
    f = f.reshape((f.size, 1))
    diag_vec = (-np.exp(f) / np.square(np.ones(f.shape) + np.exp(f)))
    return -np.diag(diag_vec.reshape((diag_vec.size, )))


def logistic_grad(f, y):
    f = f.reshape((f.size, 1))
    return -((y + np.ones(y.shape))/2 - sigmoid(f))
    # return -y / (np.exp(-y * f))


#Reassigning the parameters
d, n = x_g.shape
m = np.vectorize(lambda x: 0)
covariance_obj = model_params.cov_obj
K = covariance_obj.covariance_function
ml = covariance_obj.oracle

#Initializing covariance matrices
K_x = covariance_mat(K, x_g, x_g)
K_x_test = covariance_mat(K, x_g, x_test)
K_test_x = covariance_mat(K, x_test, x_g)
K_test = covariance_mat(K, x_test, x_test)

#Optimization
f_0 = np.zeros(y_g.shape)
res = opt.minimize(fun=(lambda f: logistic_loss(f, y_g)), x0=f_0.reshape((f_0.size,)), method='Newton-CG',
                   jac=lambda f: logistic_grad(f, y_g).reshape((f_0.size,)), hess=logistic_likelyhood_hessian, options={'disp':False})
f_opt = res['x']


def func(w):
    loss, gradient = covariance_obj.oracle(x_g, y_g, w)
    return loss

def grad(w):
    loss, gradient = covariance_obj.oracle(x_g, y_g, w)
    return gradient

# w0 = covariance_obj.get_params() + np.array([0.5, 0.5, 0.5])
# loss_1 = func(w0)
# grad_1 = grad(w0)
# loss_eps = func(w0 + np.array([1e-6, 0, 0]))
# print((loss_eps - loss_1)*1e6, grad_1)
# print(check_grad(func, grad, w0))

#Calculating the classification results on the grid
f_test = np.dot(np.dot(K_test_x, np.linalg.inv(K_x)), f_opt)
app_y_test = sigmoid(f_test.reshape((f_test.size, 1)))
app_y_test = np.sign(app_y_test - np.ones(y_test.shape) * 0.5)
# print(np.linalg.norm(app_y_test - y_test))

