import numpy as np
from scipy.special import gamma, kv
from abc import ABCMeta, abstractmethod

# General functions


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(- x))


def delta(r):
    """Delta-function"""
    if np.all(np.diag(r) == 0):
        return np.eye(r.shape[1])
    return np.zeros((r.shape))
    # return np.ones(r.shape) * ((r == 0.).astype(float))


def gaussian_noise_term(noise_variance, r):
    return noise_variance**2 * delta(r)


# def add_noise(cov_func, noise_variance):
#     def f(x, y):
#         return cov_func(x, y) + gaussian_noise_term(noise_variance, x, y)
#     return f


def covariance_mat(covariance_func, x, y):
    """Computing covariance matrix for given covariance function and point arrays"""
    # return covariance_func(x[:, :, None], y[:, None, :])
    return covariance_func(x, y)


def pairwise_distance(x, y):
    """
    Compute a matrix of pairwise distances between x and y
    :param x: array
    :param y: array
    :return: pairwise distances matrix
    """
    x_norm = np.linalg.norm(x, axis=0)[:, None]
    y_norm = np.linalg.norm(y, axis=0)[None, :]
    d = np.square(x_norm) + np.square(y_norm) - 2 * x.T.dot(y)
    d[d < 0] = 0
    # print(np.min(d))
    return np.sqrt(d)


def stationary_cov(fun):
    def wrapper(self, x, y, *args, **kwargs):
        dists = pairwise_distance(x, y)
        return fun(self, dists, *args, **kwargs)
    return wrapper

# Specific covariance functions

class CovarianceFamily:
    """This is an abstract class, representing the concept of a family of covariance functions"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def covariance_function(self, x, y, w=None):
        """
        A covariance function
        :param x: vector
        :param y: vector
        :param w: hyper-parameters vector of the covariance functions' family
        :return: the covariance between the two vectors
        """
        pass

    @staticmethod
    @abstractmethod
    def get_bounds():
        """
        :return: The bouns on the hyper-parameters
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
    def get_params(self):
        """
        A getter function for the hyper-parameters
        :param params: a vector of hyper-parameters
        :return: CovarianceFamily object
        """
        pass

    @abstractmethod
    def get_derivative_function_list(self, params):
        """
        :return: a list of functions, which produce the derivatives of the covariance matrix with respect to
        hyper-parameters except for the noise variance, when given to the covariance_matrix() function
        """
        pass

    @abstractmethod
    def covariance_derivative(self, x, y):
        """derivative wrt x"""

    def get_noise_derivative(self, points_num):
        """
        :return: the derivative of the covariance matrix w.r.t. to the noise variance.
        """
        return 2 * self.get_params()[-1] * np.eye(points_num)


class StationaryCovarianceFamily(CovarianceFamily):
    """This is an abstract class, representing the concept of a family of stationary covariance functions"""
    __metaclass__ = ABCMeta

    def covariance_function(self, x, y, w=None):
        return self.st_covariance_function(pairwise_distance(x, y), w)

    @abstractmethod
    def st_covariance_function(self, d, w=None):
        pass


class SquaredExponential(StationaryCovarianceFamily):
    """A class, representing the squared-exponential covariance functions family."""

    def __init__(self, params):
        if params.size != 3:
            raise ValueError("Wrong parameters for SquaredExponential")

        self.sigma_f = params[0]
        self.l = params[1]
        self.sigma_l = params[2]

    def get_params(self):
        return np.array([self.sigma_f, self.l, self.sigma_l])

    @staticmethod
    def get_bounds():
        return (1e-2, None), (1e-2, None), (1e-2, None)

    def set_params(self, params):
        if params.size != 3:
            raise ValueError("Wrong parameters for SquaredExponential")

        self.sigma_f = params[0]
        self.l = params[1]
        self.sigma_l = params[2]

    # @stationary_cov
    def st_covariance_function(self, r, w=None):
        if w is None:
            l = self.l
            sigma_f = self.sigma_f
            sigma_l = self.sigma_l
        else:
            sigma_f = w[0]
            l = w[1]
            sigma_l = w[2]
        return np.exp(-r**2 / (2*(l**2))) * sigma_f**2 + gaussian_noise_term(sigma_l, r)

    # @stationary_cov
    def covariance_derivative(self, x, y):
        """derivative wrt x"""
        r = pairwise_distance(x, y)
        # print(x[:, :, None] - y[:, None, :])
        # exit(0)
        return - np.exp(-r**2 / (2 * self.l**2)) * self.sigma_f**2 * 2 * \
            (x[:, :, None] - y[:, None, :]) / (2 * self.l**2)

    @stationary_cov
    def _dse_dl(self, r):
        return (np.exp(-r**2 / (2*(self.l**2))) * self.sigma_f**2) * (r**2 / (self.l ** 3))

    @stationary_cov
    def _dse_dsigmaf(self, r):
        return 2 * self.sigma_f * np.exp(-r**2 / (2*(self.l**2)))

    def get_derivative_function_list(self, params):
        se = SquaredExponential(params)
        return [se._dse_dsigmaf, se._dse_dl]


class GammaExponential(StationaryCovarianceFamily):
    """A class, representing the squared-exponential covariance functions family."""

    def __init__(self, params):
        self.sigma_f = params[0]
        self.l = params[1]
        self.gamma = params[2]
        self.sigma_l = params[3]

    def get_params(self):
        return np.array([self.sigma_f, self.l, self.gamma, self.sigma_l])

    @staticmethod
    def get_bounds():
        return (1e-2, None), (1e-2, None), (1e-2, 2), (1e-5, None)

    def set_params(self, params):
        self.sigma_f = params[0]
        self.l = params[1]
        self.gamma = params[2]
        self.sigma_l = params[3]

    def st_covariance_function(self, r, w=None):
        if w is None:
            l = self.l
            sigma_f = self.sigma_f
            g = self.gamma
            sigma_l = self.sigma_l
        else:
            sigma_f = w[0]
            l = w[1]
            g = w[2]
            sigma_l = w[3]
        return np.exp(-np.power((r / l), g)) * np.square(sigma_f) + gaussian_noise_term(sigma_l, r)

    @stationary_cov
    def _dge_dl(self, r):
        return np.exp(-(r/self.l)**self.gamma) * self.sigma_f**2 * (self.gamma * (r/self.l)**self.gamma) / self.l

    @stationary_cov
    def _dge_dsigmaf(self, r):
        return 2 * self.sigma_f * np.exp(-(r /self.l)**self.gamma)

    @stationary_cov
    def _dge_dgamma(self, r):
        loc_var = r/self.l
        loc_var_gamma = loc_var ** self.gamma
        loc_var[loc_var == 0] = 1 # A dirty hack to avoid log(0)
        res = -self.sigma_f**2 * loc_var_gamma * np.log(loc_var) * np.exp(-loc_var_gamma)
        return res

    def get_derivative_function_list(self, params):
        ge = GammaExponential(params)
        return [ge._dge_dsigmaf, ge._dge_dl, ge._dge_dgamma]


class Matern(StationaryCovarianceFamily):
    def __init__(self, params):
        if params.size != 4:
            raise ValueError("Wrong parameters for Matern")
        self.sigma_f = params[0]
        self.l = params[1]
        self.nu = params[2]
        self.sigma_l = params[3]

    def get_params(self):
        return np.array([self.sigma_f, self.l, self.nu, self.sigma_l])

    def set_params(self, params):
        if params.size != 4:
            raise ValueError("Wrong parameters for Matern")
        self.sigma_f = params[0]
        self.l = params[1]
        self.nu = params[2]
        self.sigma_l = params[3]

    def st_covariance_function(self, r, w=None):
        if w is None:
            l = self.l
            nu = self.nu
            sigma_f = self.sigma_f
            sigma_l = self.sigma_l
        else:
            sigma_f = w[0]
            l = w[1]
            nu = w[2]
            sigma_l = w[3]
        anc_var = np.sqrt(2.0 * nu) * r / l
        res = sigma_f**2 *(2.0 ** (1.0 - nu) / gamma(nu)) * (anc_var ** nu) * kv(nu, anc_var)
        res[r == 0] = sigma_f**2
        res += gaussian_noise_term(sigma_l, r)
        return res

    @staticmethod
    def get_bounds():
        return (1e-2, None), (1e-2, None), (1e-2, None), (1e-5, None)

    @stationary_cov
    def _dm_dl(self, r):
        return 1e8 * (self.st_covariance_function(r, w=(self.get_params() + np.array([0, 1e-8, 0, 0]))) -
                      self.st_covariance_function(r))
    @stationary_cov
    def _dm_dnu(self, r):
        return 1e8 * (self.st_covariance_function(r, w=(self.get_params() + np.array([0, 0, 1e-8, 0]))) -
                      self.st_covariance_function(r))

    @stationary_cov
    def _dm_dsigmaf(self, r):
        anc_var = np.sqrt(2.0 * self.nu) * r / self.l
        res = 2 * self.sigma_f * (2.0 ** (1.0 - self.nu) / gamma(self.nu)) * (anc_var ** self.nu) * kv(self.nu,
                                                                                                        anc_var)
        res[r == 0] = 2 * self.sigma_f
        return res

    def get_derivative_function_list(self, params):
        m = Matern(params)
        return [m._dm_dsigmaf, m._dm_dl, m._dm_dnu]


# class ExpScaledSquaredExponential(CovarianceFamily):
#     """A class, representing the squared-exponential covariance functions family
#     with hyper-parameters in exponential scale."""
#
#     def __init__(self, params):
#         if params.size != 3:
#             raise ValueError("Wrong parameters for SquaredExponential")
#
#         self.sigma_f = params[0]
#         self.l = params[1]
#         self.sigma_l = params[2]
#
#     def get_params(self):
#         return np.array([self.sigma_f, self.l, self.sigma_l])
#
#     @staticmethod
#     def get_bounds():
#         return (-2, 10), (-2, 10), (-5, 10)
#
#     def set_params(self, params):
#         if params.size > 3:
#             raise ValueError("Wrong parameters for SquaredExponential")
#
#         self.sigma_f = params[0]
#         self.l = params[1]
#         self.sigma_l = params[2]
#
#     def covariance_function(self, x, y, w=np.NaN):
#         if np.all(np.isnan(w)):
#             l = self.l
#             sigma_f = self.sigma_f
#             sigma_l = self.sigma_l
#         else:
#             sigma_f = w[0]
#             l = w[1]
#             sigma_l = w[2]
#         r = np.linalg.norm(x - y, axis=0)
#         return np.exp(-r**2 / (2*(np.exp(2 * l)))) * np.exp(sigma_f * 2) + gaussian_noise_term(np.exp(sigma_l), x, y)
#
#     def _dcov_dl(self, x, y):
#         r = np.linalg.norm(x - y, axis=0)
#         return (np.exp(-r**2 / (2*(np.exp(self.l*2)))) * np.exp(self.sigma_f*2)) * (r**2 / (np.exp(self.l * 3))) * np.exp(self.l)
#
#     def _dcov_dsigmaf(self, x, y):
#         r = np.linalg.norm(x - y, axis=0)
#         return np.exp(-r**2 / (2*(np.exp(2 * self.l)))) * 2 * np.exp(self.sigma_f * 2)
#
#     def get_noise_derivative(self, points_num):
#         """
#         :return: The coefficient for the identity matrix in the derivative of the covariance w.r.t. noise variance.
#         """
#         return 2 * np.exp(2 * self.sigma_l) * np.eye(points_num)
#
#     def get_derivative_function_list(self, params):
#         cov_obj = ExpScaledSquaredExponential(params)
#         return [cov_obj._dcov_dsigmaf, cov_obj._dcov_dl]


# if __name__ == '__main__':
#     alpha = 3
#     l = 0.2
#     sigma = 2
#     f = matern_cov(sigma, alpha, l)
#     x = np.array([[1], [1]])
#     y = np.array([[0], [0]])
#     print(np.linalg.norm(x - y))
#     print(f(x, y))