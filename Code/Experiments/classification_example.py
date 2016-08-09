import matplotlib.pyplot as plt
import numpy as np

from GP.covariance_functions import SquaredExponential
from GP.gaussian_process_classification import GPC
from GP.plotting import plot_class_data

data_params = np.array([1.0, 0.25, 0.05])
data_covariance_obj = SquaredExponential(data_params)
model_params = np.array([.7, .6, 0.2])
model_covariance_obj = SquaredExponential(model_params)
gp = GPC(data_covariance_obj)
num = 500
test_density = 20
dim = 2
seed = 21
ind_num = 20
method = 'vi'
maxiter = 100
max_out_iter = 5

# opts = {'mode': 'batch', 'batch_size': 200, 'maxiter': maxiter, 'verbose': True, 'step0':0.007}
# opts = {'mode': 'full', 'maxiter': maxiter, 'mydisp': True}
# opts = {'mode': 'adadelta', 'maxiter': maxiter, 'verbose': True, 'batch_size':20, 'step_rate': 0.7, 'decay': 0.8,
#         'print_freq': 10}
opts = {'bound': 'JJ', 'maxfun': 20, 'num_updates': 10, 'mydisp': True}

np.random.seed(seed)
x_tr = np.random.rand(dim, num)
if dim == 2:
    x_1 = np.linspace(0, 1, test_density)
    x_2 = np.linspace(0, 1, test_density)
    x_1, x_2 = np.meshgrid(x_1, x_2)
    x_test = np.array(list(zip(x_1.reshape(-1).tolist(), x_2.reshape(-1).tolist())))
    x_test = x_test.T
else:
    x_test = np.random.rand(dim, test_density**2)

print("Generating Data")

y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)

print("Data generated")

new_gp = GPC(model_covariance_obj, method=method, hermgauss_deg=100)

if method == 'svi':
    new_gp.fit(x_tr, y_tr, num_inputs=ind_num, optimizer_options=opts)
elif method == 'vi':
    new_gp.fit(x_tr, y_tr, num_inputs=ind_num, max_out_iter=max_out_iter, optimizer_options=opts)
elif method == 'brute':
    new_gp.fit(x_tr, y_tr)
else:
    raise ValueError("Unknown method")

print(new_gp.covariance_obj.get_params())
if method == 'brute':
    predicted_y_test = new_gp.predict(x_test, x_tr, y_tr)
elif method == 'svi' or method == 'vi':
    predicted_y_test = new_gp.predict(x_test)
    inducing_points, mean, cov = new_gp.inducing_inputs

mistake_lst = [i for i in range(len(y_test)) if predicted_y_test[i] != y_test[i]]

print("Mistakes: ", np.sum(predicted_y_test != y_test))
if dim == 2:
    plot_class_data(x_tr, y_tr, 'bo', 'mx')
    # plt.contour(np.linspace(0, 1, test_density), np.linspace(0, 1, test_density),
    #             y_test.reshape((test_density, test_density)), levels=[0], colors=['g'])
    plt.contour(np.linspace(0, 1, test_density), np.linspace(0, 1, test_density),
                predicted_y_test.reshape((test_density, test_density)), levels=[0], colors=['y'])
    if method != 'brute':
        plt.plot(inducing_points[0, :], inducing_points[1, :], 'ro', markersize=10)
    plt.show()
