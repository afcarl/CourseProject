import numpy as np
import matplotlib.pyplot as plt
from covariance_functions import covariance_mat
from gp_class_data import x_g, y_g, plot_data, x_test, y_test, sigmoid
from class_parameters import model_params, common_params
import scipy.optimize as opt


def logistic_loss(f, y):
    """Logistic loss function"""
    f = f.reshape((f.size, 1))
    return (- np.sum(np.log(np.exp(-y * f) + np.ones(y.shape))))


def logistic_likelyhood_hessian(f):
    """Hessian of the p(y|f)"""
    f = f.reshape((f.size, 1))
    diag_vec = (-np.exp(f) / np.square(np.ones(f.shape) + np.exp(f)))
    return np.diag(diag_vec.reshape((diag_vec.size, )))

def logistic_grad(f, y):
    f = f.reshape((f.size, 1))
    return (y / ( np.exp(-y * f)))


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
                   jac=lambda f: logistic_grad(f, y_g).reshape((f_0.size,)), hess=logistic_likelyhood_hessian, options={'maxiter': 10})
f_opt = res['x']

#Calculating the classification results on the grid
f_test = np.dot(np.dot(K_test_x, np.linalg.inv(K_x)), f_opt)
y_test = sigmoid(f_test.reshape((f_test.size, 1)))
app_y_test = np.sign(y_test - np.ones(y_test.shape) * 0.5)

print(np.linalg.norm(app_y_test - y_test))