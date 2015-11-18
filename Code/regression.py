import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, cross_validation
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from gaussian_process import GaussianProcess
from plotting import gp_plot_reg_data, gp_plot_class_data
from covariance_functions import SquaredExponential, GammaExponential, Matern

data_params = np.array([1.1, 0.1, 0.1])
data_covariance_obj = SquaredExponential(data_params)
gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'reg')
num = 200
test_num = 1000
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

# model_params = np.array([1., 0.7, 0.2])
# model_covariance_obj = SquaredExponential(model_params)
# new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
# new_gp.find_hyper_parameters(x_tr, y_tr, max_iter=30)
# predicted_y_test, high, low = new_gp.predict(x_test, x_tr, y_tr)

model_params = np.array([1., 0.2, 0.2])
model_covariance_obj = SquaredExponential(model_params)
new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
inducing_points, mean, cov, _, _, _ = new_gp.reg_find_inducing_inputs(x_tr, y_tr, 30, max_iter=30)
predicted_y_test, high, low = new_gp.predict(x_test, x_tr, y_tr)

# means = KMeans(n_clusters=6)
# means.fit(x_tr.T)
# inducing_points = means.cluster_centers_.T
# targets = []
# for i in range(inducing_points.shape[1]):
#     nbrs = NearestNeighbors(n_neighbors=1).fit(x_tr.T)
#     mean = inducing_points[:, i]
#     _, indices = nbrs.kneighbors(mean[:, None])
#     targets.append(y_tr[indices][0,0,0])
#     print(y_tr[indices][0,0,0])
# targets = np.array(targets)[:, None]
# print(targets.shape)
# small_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
# _, _, fun_lst = small_gp.find_hyper_parameters(inducing_points, targets, max_iter=30)
# print("Evidence: ", fun_lst[-1])

# predicted_y_test, high, low = small_gp.predict(x_test, x_tr, y_tr)


# print(inducing_points)
# print(mean.shape)
# print(cov.shape)
# predicted_y_test, high, low = new_gp.reg_inducing_points_predict(inducing_points, mean, cov, x_test)

print(new_gp.covariance_obj.get_params())
print(np.linalg.norm(predicted_y_test - y_test)/y_test.size)

if dim == 1:
    gp_plot_reg_data(x_tr, y_tr, 'yo')
    gp_plot_reg_data(x_test, predicted_y_test, 'b')
    gp_plot_reg_data(x_test, low, 'g-')
    gp_plot_reg_data(x_test, high, 'r-')
    gp_plot_reg_data(x_test, y_test, 'y-')
    gp_plot_reg_data(inducing_points, mean, 'ro', markersize=12)

if dim == 1:
    plt.show()

