import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from gaussian_process import GaussianProcess, gp_plot_reg_data, gp_plot_class_data
from covariance_functions import CovarianceFamily, SquaredExponential

data_params = np.array([2.0, 0.1, 0.1])
data_covariance_obj = SquaredExponential(data_params)
gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'reg')
num = 10
test_num = 100
dim = 1
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

model_params = np.array([1.,  0.1,  0.05])
model_covariance_obj = SquaredExponential(model_params)
new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
new_gp._reg_find_hyper_parameters(x_tr, y_tr)
print(new_gp.covariance_obj.get_params())
predicted_y_test = new_gp._reg_predict(x_test, x_tr, y_tr)

gp_plot_reg_data(x_tr, y_tr, 'rx')
gp_plot_reg_data(x_test, predicted_y_test, 'b')
gp_plot_reg_data(x_test, y_test, 'bx')

clf = svm.SVR(C=100, epsilon=0.2)
clf.fit(x_tr.T, y_tr.reshape((y_tr.size,)))
svm_y_test = clf.predict(x_test.T)
# gp_plot_reg_data(x_test, svm_y_test, 'y')
print(np.linalg.norm(y_test - svm_y_test))
print(np.linalg.norm(y_test - predicted_y_test))


plt.show()