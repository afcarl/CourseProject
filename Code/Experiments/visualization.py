import numpy as np
from matplotlib import pyplot as plt

from GP.covariance_functions import SquaredExponential
from GP.gaussian_process_regression import GPR
from GP.plotting import gp_plot_reg_data

data_params = np.array([1.1, 0.3, 0.1])
data_covariance_obj =  SquaredExponential(data_params)
# model_params = np.array([10.6, 5.2, 0.1])
model_params = np.array([1.5, 0.15, 0.1])
model_covariance_obj = SquaredExponential(model_params)
gp = GPR(data_covariance_obj)
num = 700
test_num = 100
dim = 1
seed = 22
method = 'brute'  # possible methods: 'brute', 'vi', 'means', 'svi'
parametrization = 'natural'  # possible parametrizations for svi method: cholesky, natural
ind_inputs_num = 5
max_iter = 100
lbfgsb_options = {'maxiter': max_iter, 'disp': False}

np.random.seed(seed)
x_tr = np.random.rand(dim, num)
if dim == 1:
    x_test = np.linspace(0, 1, test_num)
    x_test = x_test.reshape(1, test_num)
else:
    x_test = np.random.rand(dim, test_num)
y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)


data_points = []
data_targets = []

fig = plt.figure()
gp_plot_reg_data(x_test, y_test, 'y-')

means_gp = GPR(model_covariance_obj, method='means')
means_gp.fit(x_tr, y_tr, num_inputs=ind_inputs_num, optimizer_options=lbfgsb_options)
print(model_covariance_obj.get_params())
means_inducing_points, means_mean, means_cov = means_gp.inducing_inputs
means_y_test, means_high, means_low = means_gp.predict(x_test)

def onclick(event):
    plt.close('all')

    point_x, point_y = event.xdata, event.ydata
    data_points.append(point_x)
    data_targets.append(point_y)

    x_tr = np.array(data_points).reshape(-1)[None, :]
    y_tr = np.array(data_targets)
    new_gp = GPR(model_covariance_obj, method=method)
    # new_gp.fit(x_tr, y_tr, max_iter=max_iter)
    print(new_gp.covariance_obj.get_params())
    predicted_y_test, high, low = new_gp.predict(x_test, x_tr, y_tr)

    fig = plt.figure()
    gp_plot_reg_data(x_tr, y_tr, 'yo')
    gp_plot_reg_data(x_test, predicted_y_test, 'b')
    gp_plot_reg_data(x_test, means_y_test, '--b')
    gp_plot_reg_data(means_inducing_points, means_mean, 'bo', markersize=12)
    gp_plot_reg_data(x_test, means_low, '--g')
    gp_plot_reg_data(x_test, means_high, '--r')
    gp_plot_reg_data(x_test, low, 'g-')
    gp_plot_reg_data(x_test, high, 'r-')
    gp_plot_reg_data(x_test, y_test, 'y-')
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.ylim(-2, 2)
    plt.xlim(0, 1)
    plt.show()

gp_plot_reg_data(x_tr, y_tr, 'ro')
gp_plot_reg_data(x_test, means_low, '--g')
gp_plot_reg_data(x_test, means_high, '--r')
gp_plot_reg_data(x_test, means_y_test, '--b')
gp_plot_reg_data(means_inducing_points, means_mean, 'bo', markersize=12)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.xlim(0, 1)
plt.ylim(-2, 2)
plt.show()