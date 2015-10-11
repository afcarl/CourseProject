import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation

from gaussian_process import GaussianProcess, gp_plot_reg_data, gp_plot_class_data
from covariance_functions import CovarianceFamily, SquaredExponential

data_params = np.array([2.0, 0.1, 0.1])
data_covariance_obj = SquaredExponential(data_params)
gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'reg')
num = 20
test_num = 102
dim = 1
seed = 21

np.random.seed(seed)
x_tr = np.random.rand(dim, num)
if dim == 1:
    x_test = np.linspace(0, 1, test_num)
    x_test = x_test.reshape(1, test_num)
else:
    x_test = np.random.rand(dim, test_num)
y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)

model_params = np.array([1.,  0.1,  0.05])
model_covariance_obj = SquaredExponential(model_params)
new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
new_gp.find_hyper_parameters(x_tr, y_tr)
print(new_gp.covariance_obj.get_params())
predicted_y_test = new_gp.predict(x_test, x_tr, y_tr)

# print("Training SVM...")
# n_samples = x_tr.shape[1]
# max_score = -np.inf
# c_max, eps_max = 0, 0
# for c, eps in list(zip([1, 10, 1000, 10000], [10, 1, 0.5, 0.1, 0.01])):
#     clf = svm.SVR(C=c, epsilon=eps)
#     cv = cross_validation.ShuffleSplit(n_samples, n_iter=3, test_size=0.3, random_state=0)
#     scores = cross_validation.cross_val_score(clf, x_tr.T, y_tr.reshape((y_tr.size,)), cv=5)
#     if scores.mean() > max_score:
#         max_score = scores.mean()
#         c_max, eps_max = c, eps
# clf = svm.SVR(C=c_max, epsilon=eps_max)
# clf.fit(x_tr.T, y_tr.reshape((y_tr.size,)))
# svm_y_test = clf.predict(x_test.T)

if dim == 1:
    gp_plot_reg_data(x_tr, y_tr, 'rx')
    gp_plot_reg_data(x_test, predicted_y_test, 'b')
    gp_plot_reg_data(x_test, y_test, 'bx')
    # gp_plot_reg_data(x_test, svm_y_test, 'y')

# print(np.linalg.norm(y_test - svm_y_test))
print(np.linalg.norm(y_test - predicted_y_test))

plt.show()
