import numpy as np

from GP.covariance_functions import SquaredExponential
from GP.gaussian_process_regression import GPR

data_params = np.array([1.1, 0.3, 0.1])
data_covariance_obj = SquaredExponential(data_params)
# model_params = np.array([10.1, 1.2, 0.1])
model_params = np.array([1.2, 0.6, 0.1])
# model_params = np.array([0.6, 0.2, 0.1])
model_covariance_obj = SquaredExponential(model_params)
gp = GPR(data_covariance_obj)
num = 500
test_num = 200
dim = 3
seed = 21
method = 'means'  # possible methods: 'brute', 'vi', 'means', 'svi'
ind_inputs_num = 5
max_iter = 200

# Generating data points
np.random.seed(seed)
x_tr = np.random.rand(dim, num)
if dim == 1:
    x_test = np.linspace(0, 1, test_num)
    x_test = x_test.reshape(1, test_num)
else:
    x_test = np.random.rand(dim, test_num)
y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)

# Cholesky parametrization

parametrization = 'cholesky'
for optimizer, color in zip(['SAG', 'FG', 'L-BFGS-B'], '-ro, -bo, -go'):
    model_covariance_obj = SquaredExponential(np.copy(model_params))
    new_gp = GPR(model_covariance_obj, method=method, parametrization=parametrization, optimizer=optimizer)
    res = new_gp.fit(x_tr, y_tr, num_inputs=ind_inputs_num, max_iter=max_iter)

    # inducing_points, mean, cov = new_gp.inducing_inputs
    # predicted_y_test, high, low = new_gp.predict(x_test)


    print(new_gp.covariance_obj.get_params())
    print(np.linalg.norm(predicted_y_test - y_test)**2/y_test.size)



