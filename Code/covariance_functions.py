import numpy as np
from scipy.optimize import check_grad
from scipy.special import gamma, kv
from abc import ABCMeta, abstractmethod

# General functions


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(- x))


def delta(x, y):
    """Delta-function"""
    if np.all(x[:, :, 0] == y[:, 0, :]):
        return np.eye(x.shape[1])
    return np.zeros((x.shape[1], y.shape[1]))


def gaussian_noise_term(noise_variance, x, y):
    return noise_variance**2 * delta(x, y)


def add_noise(cov_func, noise_variance):
    def f(x, y):
        return cov_func(x, y) + gaussian_noise_term(noise_variance, x, y)
    return f


def covariance_mat(covariance_func, x, y):
    """Computing covariance matrix for given covariance function and point arrays"""
    return covariance_func(x[:, :, None], y[:, None, :])

# Specific covariance functions


class CovarianceFamily:
    __metaclass__ = ABCMeta
    """This is an abstract class, representing the concept of a family of covariance functions"""

    @abstractmethod
    def covariance_function(self, x, y, w=np.NaN):
        """
        A covariance function
        :param x: vector
        :param y: vector
        :param w: hyper-parameters vector of the covariance functions' family
        :return: the covariance between the two vectors
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        :return: The hyper-parameters vector of the CovarianceFamily object
        """
        pass

    @abstractmethod
    def set_params(self, params):
        """
        A setter function for the hyper-parameters
        :param params: a vector of hyper-parameters
        :return: CovarianceFamily object
        """
        pass

    @abstractmethod
    def get_derivative_function_list(self, params):
        """
        :return: a list of functions, which produce the derivatives of the covariance matrix with respect to
        hyper-parameters, when given to the covariance_matrix() function
        """
        pass


class SquaredExponential(CovarianceFamily):
    """A class, representing the squared-exponential covariance functions family."""

    def __init__(self, params):
        if params.size > 3:
            raise ValueError("Wrong parameters for SquaredExponential")

        self.sigma_f = params[0]
        self.l = params[1]
        self.sigma_l = params[2]

    def get_params(self):
        return np.array([self.sigma_f, self.l, self.sigma_l])

    def set_params(self, params):
        if params.size > 3:
            raise ValueError("Wrong parameters for SquaredExponential")

        self.sigma_f = params[0]
        self.l = params[1]
        self.sigma_l = params[2]

    def covariance_function(self, x, y, w=np.NaN):
        if np.all(np.isnan(w)):
            l = self.l
            sigma_f = self.sigma_f
            sigma_l = self.sigma_l
        else:
            sigma_f = w[0]
            l = w[1]
            sigma_l = w[2]
        r = np.linalg.norm(x - y, axis=0)
        return np.exp(-r**2 / (2*(l**2))) * sigma_f**2 + gaussian_noise_term(sigma_l, x, y)

    def _dse_dl(self, x, y):
        r = np.linalg.norm(x - y, axis=0)
        return (np.exp(-r**2 / (2*(self.l**2))) * self.sigma_f**2) * (r**2 / (self.l ** 3))

    def _dse_dsigmaf(self, x, y):
        r = np.linalg.norm(x - y, axis=0)
        return 2 * self.sigma_f * np.exp(-r**2 / (2*(self.l**2)))

    def get_derivative_function_list(self, params):
        se = SquaredExponential(params)
        return [se._dse_dsigmaf, se._dse_dl]


class GammaExponential(CovarianceFamily):
    """A class, representing the squared-exponential covariance functions family."""

    def __init__(self, sigma_f, l, gamma, noise):
        self.sigma_f, self.l, self.gamma, self.sigma_l = sigma_f, l, gamma, noise

    def get_params(self):
        return np.array([self.sigma_f, self.l, self.gamma, self.sigma_l])

    def set_params(self, params):
        self.sigma_f = params[0]
        self.l = params[1]
        self.gamma = params[2]
        self.sigma_l = params[3]

    def covariance_function(self, x, y, w=np.NaN):
        if np.all(np.isnan(w)):
            l = self.l
            sigma_f = self.sigma_f
            gamma = self.gamma
            sigma_l = self.sigma_l
        else:
            sigma_f = w[0]
            l = w[1]
            gamma = w[2]
            sigma_l = w[3]
        r = np.linalg.norm(x - y, axis=0)
        return np.exp(-(r / l)**gamma) * sigma_f**2 + gaussian_noise_term(sigma_l, x, y)


def matern_cov(nu, l):
    """Matern covariance function"""
    def f(x, y):
        r = np.linalg.norm(x - y)
        if r == 0:
            return 1.0
        anc_var = np.sqrt(2.0 * np.pi) * r / l
        return (2.0 ** (1.0 - nu) / gamma(nu)) * (anc_var ** nu) * kv(nu, anc_var)
    return f


def gamma_exponential_cov(gamma, l):
    """Gamma-exponential covariance function, 0 < gamma ≤ 2"""
    def f(x, y):
        r = np.linalg.norm(x - y)
        return np.exp(-(r / l) ** gamma)
    return f


def rational_quadratic_cov(alpha, l):
    """Rational-quadratic covariance function"""
    def f(x, y):
        r = np.linalg.norm(x - y)
        return (1 + (np.square(r) / (2 * alpha * np.square(l))))**(-alpha)
    return f

if __name__ == '__main__':
    sigma_f = 1.2
    sigma_l = 0.1
    l = 0.2
    gamma = 0.8

    w0 = np.array([sigma_f, l, sigma_l])
    se = SquaredExponential(sigma_f, l, sigma_l)
    x = np.array([[1., 2., 3.], [4., 5., 6.]])
    y = np.array([0., 1., 0.])
    # print (covariance_mat(ge.covariance_function, x, x))
    def func(w):
        loss, gradient = se.oracle(x, y, w)
        return loss

    def grad(w):
        loss, gradient = se.oracle(x, y, w)
        return gradient

    print(check_grad(func, grad, w0))
