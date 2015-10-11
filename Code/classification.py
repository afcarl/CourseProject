import numpy as np
import matplotlib.pyplot as plt
import time

from gaussian_process import GaussianProcess, gp_plot_reg_data, gp_plot_class_data
from covariance_functions import CovarianceFamily, SquaredExponential

data_params = np.array([1.0, 0.25, 0.05])
data_covariance_obj = SquaredExponential(data_params)
gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'class')
num = 200
test_density = 50
dim = 2
# x_tr, y_tr = gp.generate_data(dim, num+test_num, seed=21)
# x_test = x_tr[:, num:]
# y_test = y_tr[num:]
# x_tr = x_tr[:, :num]
# y_tr = y_tr[:num]
seed = 21

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

y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)

model_params = np.array([2., 0.7, 0.2])
model_covariance_obj = SquaredExponential(model_params)
new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'class')

start_time = time.time()
new_gp.find_hyper_parameters(x_tr, y_tr, max_iter=100)
timing = time.time() - start_time
print(timing)

print(new_gp.covariance_obj.get_params())
predicted_y_test = new_gp.predict(x_test, x_tr, y_tr)
gp_plot_class_data(x_tr, y_tr, 'bo', 'ro')
# gp_plot_class_data(x_test, predicted_y_test, 'bx', 'rx')
# mistake_lst = [i for i in range(len(y_test)) if predicted_y_test[i] != y_test[i]]
# gp_plot_class_data((x_test[:, mistake_lst]),
#                    y_test[mistake_lst, :], 'gs', 'ys')
print("Mistakes: ", np.sum(predicted_y_test != y_test))
plt.contour(np.linspace(0, 1, test_density), np.linspace(0, 1, test_density),
            y_test.reshape((test_density, test_density)), levels=[0], colors=['g'])
plt.contour(np.linspace(0, 1, test_density), np.linspace(0, 1, test_density),
            predicted_y_test.reshape((test_density, test_density)), levels=[0], colors=['y'])
plt.show()
