import numpy as np
import matplotlib.pyplot as plt
from covariance_functions import covariance_mat, sigmoid
from class_parameters import data_params, common_params

def plot_data(x, y, color1, color2):
    loc_y = y.reshape((y.shape[0],))
    plt.plot(x[0, loc_y == 1], x[1, loc_y == 1], color1)
    plt.plot(x[0, loc_y == -1], x[1, loc_y == -1],color2)


def sample(mean_func, cov_func, x):
    """returns a sample of a gaussian process for given mean and covariance at given points"""
    cov_mat = covariance_mat(cov_func, x, x)
    m_v = mean_func(x)
    mean_vector = m_v.reshape((m_v.size,))
    np.random.seed(data_params.data_seed)
    y = np.random.multivariate_normal(mean_vector, cov_mat)
    return y


#Reassigning the parameters
d, n, t = common_params.d, common_params.n, common_params.t
m = lambda x: np.zeros(x.shape[1])
covariance_obj = data_params.cov_obj
K = (covariance_obj).covariance_function

#Producing data
np.random.seed(data_params.data_seed)
x_g = np.random.rand(d, n)
y_g = sample(m, K, x_g)
y_g = y_g.reshape((y_g.size, 1))
y_g = sigmoid(y_g.reshape((y_g.size, 1)))
y_g = np.sign(y_g - np.ones(y_g.shape) * 0.5)

x_test = np.random.rand(d, t)
y_test = sample(m, K, x_test)
y_test = sigmoid(y_test.reshape((y_test.size, 1)))
y_test = np.sign(y_test - np.ones(y_test.shape) * 0.5)


if __name__ == "__main__":
    plot_data(x_g, y_g, 'bx', 'ro')
    plt.show()
