import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import r2_score
from GP.gaussian_process_regression import GPR
from GP.plotting import gp_plot_reg_data
from GP.covariance_functions import SquaredExponential
from matplotlib.mlab import griddata

data_params = np.array([1.0, 0.3, 0.1])
data_covariance_obj = SquaredExponential(data_params)
model_params = np.array([5.1, 0.3, 0.1])
model_covariance_obj = SquaredExponential(model_params)
gp = GPR(data_covariance_obj)
num = 1000
test_num = 500
dim = 5
seed = 10
ind_inputs_num = 200
max_iter = 100
batch_size = 100

method = 'means'  # possible methods: 'brute', 'vi', 'means', 'svi'
parametrization = 'cholesky'  # possible parametrizations for svi method: cholesky, natural
optimizer = 'L-BFGS-B' # possible optimizers: 'SAG', 'FG', 'L-BFGS-B'

sag_options = {'maxiter':max_iter, 'batch_size': batch_size, 'print_freq': 100}
fg_options = {'maxiter':max_iter, 'print_freq': 100}
lbfgsb_options = {'maxiter': max_iter, 'disp': False}
sg_options = {'maxiter':max_iter, 'batch_size': batch_size, 'print_freq': 100, 'step0': 5e-3, 'gamma': 0.55}

# Generating data points
np.random.seed(seed)
x_tr = np.random.rand(dim, num)
if dim == 1:
    x_test = np.linspace(0, 1, test_num)
    x_test = x_test.reshape(1, test_num)
else:
    # x_test = np.zeros((dim, test_num))
    # x_test[:, :] = np.linspace(0, 1, test_num)
    # for i in range(dim):
    #     x_test[i, :] = np.linspace(0, 1, test_num / dim)
    x_test = np.random.rand(dim, test_num)

y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)

if method == 'brute':
    new_gp = GPR(model_covariance_obj)
    res = new_gp.fit(x_tr, y_tr, max_iter=max_iter)
    predicted_y_test, high, low = new_gp.predict(x_test, x_tr, y_tr)

elif method == 'means' or method == 'vi':
    model_covariance_obj = SquaredExponential(model_params)
    new_gp = GPR(model_covariance_obj, method=method)
    res = new_gp.fit(x_tr, y_tr, num_inputs=ind_inputs_num, max_iter=max_iter)
    inducing_points, mean, cov = new_gp.inducing_inputs
    predicted_y_test, high, low = new_gp.predict(x_test)

elif method == 'svi':
    model_covariance_obj = SquaredExponential(model_params)
    new_gp = GPR(model_covariance_obj, method=method, parametrization=parametrization, optimizer=optimizer)
    res = new_gp.fit(x_tr, y_tr, num_inputs=ind_inputs_num, optimizer_options=lbfgsb_options)
    inducing_points, mean, cov = new_gp.inducing_inputs
    predicted_y_test, high, low = new_gp.predict(x_test)

print(new_gp.covariance_obj.get_params())
print(r2_score(y_test, predicted_y_test))

if dim == 1:
    gp_plot_reg_data(x_tr, y_tr, 'yo')
    gp_plot_reg_data(x_test, predicted_y_test, 'b')
    gp_plot_reg_data(x_test, low, 'g-')
    gp_plot_reg_data(x_test, high, 'r-')
    gp_plot_reg_data(x_test, y_test, 'y-')
    if method != 'brute':
        gp_plot_reg_data(inducing_points, mean, 'ro', markersize=12)
    plt.show()

elif dim == 2:
    xi = np.linspace(0, 1, 100)
    yi = np.linspace(0, 1, 100)
    zi = griddata(x_tr[0, :], x_tr[1, :], y_tr[:, 0], xi, yi, interp='linear')
    plt.plot(x_tr[0], x_tr[1], 'yo')
    if method != 'brute':
        gp_plot_reg_data(inducing_points[0, :], inducing_points[1, :], 'ro', markersize=12)

    plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow,
                      vmax=abs(zi).max(), vmin=-abs(zi).max())
    plt.colorbar()
    plt.show()
# else:
#     x_test[1:, :] = 0
#     print(x_test[:, 0])
#     predicted_y_test, high, low = new_gp.predict(x_test)
#     gp_plot_reg_data(x_test[0, :][:, None], predicted_y_test, 'bo')
#     gp_plot_reg_data(x_test[0, :][:, None], y_test, 'yo')
#     plt.show()