import numpy as np
import matplotlib.pyplot as plt
import time

from gaussian_process import GaussianProcess, gp_plot_reg_data, gp_plot_class_data
from covariance_functions import CovarianceFamily, SquaredExponential

data_params = np.array([1.0, 0.25, 0.05])
data_covariance_obj = SquaredExponential(data_params)
gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'class')
num = 500
test_num = 1000
dim = 2
x_tr, y_tr = gp.generate_data(dim, num+test_num, seed=21)
x_test = x_tr[:, num:]
y_test = y_tr[num:]
x_tr = x_tr[:, :num]
y_tr = y_tr[:num]
if dim == 1:
    x_lst, y_lst = x_test.reshape(x_test.size).tolist(), y_test.reshape(y_test.size).tolist()
    lst = list(zip(x_lst, y_lst))
    lst = sorted(lst, key=lambda x: x[0])
    x_lst = [elem[0] for elem in lst]
    y_lst = [elem[1] for elem in lst]
    x_test = np.array(x_lst).reshape((1, x_test.size))
    y_test = np.array(y_lst).reshape((y_test.size, 1))
model_params = np.array([2., 0.7, 0.2])
model_covariance_obj = SquaredExponential(model_params)
new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'class')

start_time = time.time()
new_gp._class_find_hyper_parameters(x_tr, y_tr, max_iter=100)
timing = time.time() - start_time
print(timing)

print(new_gp.covariance_obj.get_params())
predicted_y_test = new_gp._class_predict(x_test, x_tr, y_tr)
print(new_gp.covariance_obj.get_params())
gp_plot_class_data(x_tr, y_tr, 'bo', 'ro')
gp_plot_class_data(x_test, predicted_y_test, 'bx', 'rx')
mistake_lst = [i for i in range(len(y_test)) if predicted_y_test[i] != y_test[i]]
gp_plot_class_data((x_test[:, mistake_lst]),
                   y_test[mistake_lst, :], 'gs', 'ys')
print("Mistakes: ", np.sum(predicted_y_test != y_test))
plt.show()