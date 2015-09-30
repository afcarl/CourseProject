import numpy as np
import matplotlib.pyplot as plt
from covariance_functions import covariance_mat
from reg_parameters import data_params, common_params


def plot_data(x, y, color):
    x1 = x.reshape((x.size, ))
    y1 = y.reshape((y.size, ))
    plt.plot(x1, y1, color)


def sample(mean_func, cov_func, x):
    """returns a sample of a gaussian process for given mean and covariance at given points"""
    cov_mat = covariance_mat(cov_func, x, x)
    m_v = mean_func(x)
    mean_vector = m_v.reshape((m_v.size,))
    np.random.seed(data_params.data_seed)
    y = np.random.multivariate_normal(mean_vector, cov_mat)
    return y


#Reassigning the parameters
d, n = common_params.d, common_params.n
m = lambda x: np.zeros(x.shape[1])
covariance_obj = data_params.cov_obj
K = (covariance_obj).covariance_function

#Producing data
np.random.seed(data_params.data_seed)
x_g = np.random.rand(d, n)
y_g = sample(m, K, x_g)
y_g = y_g.reshape((y_g.size, 1))
x_test = np.random.rand(d, n)
y_test = sample(m, K, x_test)

if __name__ == "__main__":
    plot_data(x_g, y_g, 'x')
    plt.show()
