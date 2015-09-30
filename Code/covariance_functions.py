import numpy as np
import math
from scipy.optimize import check_grad
from scipy.special import gamma, kv
from abc import ABCMeta, abstractmethod

# General functions


def delta(x, y):
    """Delta-function"""
    return (x == y).all()


def gaussian_noise_term(noise_variance, x, y):
    return noise_variance**2 * delta(x, y)


def add_noise(cov_func, noise_variance):
    def f(x, y):
        return cov_func(x, y) + gaussian_noise_term(noise_variance, x, y)
    return f


def covariance_mat(covariance_func, x, y):
    """Computing covariance matrix for given covariance function and point arrays"""
    mat = np.zeros((x.shape[1], y.shape[1]))
    for i in range(0, x.shape[1]):
        for j in range(0, y.shape[1]):
            mat[i, j] = covariance_func(x[:, i], y[:, j])
    return mat

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
    def oracle(self, x, y, w=np.NaN):
        """
        Oracle function for marginal likelihood for gp-regression
        :param x: an array of data-points
        :param y: a vector of target values
        :param w: a vector of covariance functions' family hyper-parameters
        :return: a tuple (the marginal likelihood, marginal likelihood gradient with respect to hyperparameters)
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

    @staticmethod
    def _get_ml(y, k_inv):
        """
        :param y: target value vector
        :param k_inv: inverse covariance matrix at data points
        :return: marginal likelihood
        """
        n = y.size
        return -((y.T.dot(k_inv)).dot(y) - np.log(np.linalg.det(k_inv)) + n * np.log(2 * math.pi))/2

    @staticmethod
    def _get_ml_partial_derivative(y, k_inv, dk_dtheta_mat):
        """
        :param y: target value vector
        :param k_inv: inverse covariance matrix at data points
        :param dk_dtheta_mat: the matrix of partial derivatives of the covariance function at data points
        with respect to hyper-parameter theta
        :return: marginal likelihood partial derivative with respect to hyper-parameter theta
        """
        return ((((y.T.dot(k_inv)).dot(dk_dtheta_mat)).dot(k_inv)).dot(y) - np.trace(k_inv.dot(dk_dtheta_mat))) / 2

    def _get_k_inv(self, x):
        """
        :param x: data points array
        :return: inverse covariance matrix at data points
        """
        return np.linalg.inv(covariance_mat(self.covariance_function, x, x))


class SquaredExponential(CovarianceFamily):
    """A class, representing the squared-exponential covariance functions family."""

    def __init__(self, sigma_f, l, noise):
        self.sigma_f, self.l, self.sigma_l = sigma_f, l, noise

    def get_params(self):
        return np.array([self.sigma_f, self.l, self.sigma_l])

    def set_params(self, params):
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
        r = np.linalg.norm(x - y)
        return np.exp(-np.square(r) / (2*(l**2))) * sigma_f**2 + gaussian_noise_term(sigma_l, x, y)

    def oracle(self, X, y, w=np.NaN):
        if np.all(np.isnan(w)):
            l = self.l
            sigma_f = self.sigma_f
            sigma_l = self.sigma_l
        else:
            sigma_f = w[0]
            l = w[1]
            sigma_l = w[2]

        n = y.size
        se = SquaredExponential(sigma_f, l, sigma_l)
        k_inv = se._get_k_inv(x=X)
        ml = se._get_ml(y, k_inv)

        def d_se_dl(x, y):
            r = np.linalg.norm(x - y)
            return (np.exp(-np.square(r) / (2*(l**2))) * sigma_f**2) * (r / (l ** 3))

        def d_se_dsigmaf(x, y):
            r = np.linalg.norm(x - y)
            return 2 * sigma_f * np.exp(-np.square(r) / (2*(l**2)))

        dk_dl_mat = covariance_mat(d_se_dl, X, X)
        dk_dsigmaf_mat = covariance_mat(d_se_dsigmaf, X, X)
        dk_dsigmal_mat = 2 * sigma_l * np.eye(n)
        g_1 = se._get_ml_partial_derivative(y, k_inv, dk_dsigmaf_mat)
        g_2 = se._get_ml_partial_derivative(y, k_inv, dk_dl_mat)
        g_3 = se._get_ml_partial_derivative(y, k_inv, dk_dsigmal_mat)
        return -ml, -np.array([g_1, g_2, g_3]).reshape(3,)


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
        r = np.linalg.norm(x - y)
        return np.exp(-(r / l)**gamma) * sigma_f**2 + gaussian_noise_term(sigma_l, x, y)

    def oracle(self, X, y, w=np.NaN):
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

        n = y.size
        ge = GammaExponential(sigma_f, l, gamma, sigma_l)
        k_inv = ge._get_k_inv(x=X)
        ml = ge._get_ml(y, k_inv)

        def d_ge_dl(x, y):
            r = np.linalg.norm(x - y)
            return sigma_f**2 * (r / l)**gamma * np.exp(-(r/l)**gamma) / l

        def d_ge_dsigmaf(x, y):
            r = np.linalg.norm(x - y)
            return sigma_f * 2 * np.exp(-(r / l)**gamma)

        def d_ge_dgamma(x, y):
            r = np.linalg.norm(x - y)
            if r == 0:
                return 0
            return - sigma_f**2 * (r/l)**gamma * np.exp(-(r/l)**gamma) * np.log(r/l)

        dk_dl_mat = covariance_mat(d_ge_dl, X, X)
        dk_dsigmaf_mat = covariance_mat(d_ge_dsigmaf, X, X)
        dk_dgamma_mat = covariance_mat(d_ge_dgamma, X, X)
        dk_dsigmal_mat = 2 * sigma_l * np.eye(n)
        g_1 = ge._get_ml_partial_derivative(y, k_inv, dk_dl_mat)
        g_2 = ge._get_ml_partial_derivative(y, k_inv, dk_dsigmaf_mat)
        g_3 = ge._get_ml_partial_derivative(y, k_inv, dk_dgamma_mat)
        g_4 = ge._get_ml_partial_derivative(y, k_inv, dk_dsigmal_mat)
        return -ml, -np.array([g_2, g_1, g_3, g_4]).reshape(4,)

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
    sigma_l = 0.5
    l = 0.2
    gamma = 0.8

    w0 = np.array([sigma_f, l, gamma, sigma_l])
    ge = GammaExponential(sigma_f, l, gamma, sigma_l)
    x = np.array([[1., 2., 3.]])
    y = np.array([0., 1., 0.])

    def func(w):
        loss, gradient = ge.oracle(x, y, w)
        return loss

    def grad(w):
        loss, gradient = ge.oracle(x, y, w)
        return gradient

    # print(check_grad(func, grad, w0))
