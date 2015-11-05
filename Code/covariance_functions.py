import numpy as np
from scipy.optimize import check_grad
from scipy.special import gamma, kv
from abc import ABCMeta, abstractmethod
from scipy import special

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
    def get_bounds(self):
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

    def get_noise_derivative(self, points_num):
        """
        :return: the derivative of the covariance matrix w.r.t. to the noise variance.
        """
        return 2 * self.get_params()[-1] * np.eye(points_num)


class SquaredExponential(CovarianceFamily):
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
        return (1e-2, None), (1e-2, None), (1e-5, None)

    def set_params(self, params):
        if params.size != 3:
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

    def covariance_function(self, x, y, w=None):
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
        r = np.linalg.norm(x - y, axis=0)
        return np.exp(-np.power((r / l), g)) * np.square(sigma_f) + gaussian_noise_term(sigma_l, x, y)

    def _dge_dl(self, x, y):
        r = np.linalg.norm(x - y, axis=0)
        return np.exp(-(r/self.l)**self.gamma) * self.sigma_f**2 * (self.gamma * (r/self.l)**self.gamma) / self.l

    def _dge_dsigmaf(self, x, y):
        r = np.linalg.norm(x - y, axis=0)
        return 2 * self.sigma_f * np.exp(-(r /self.l)**self.gamma)

    def _dge_dgamma(self, x, y):
        r = np.linalg.norm(x - y, axis=0)
        loc_var = r/self.l
        loc_var_gamma = loc_var ** self.gamma
        loc_var[loc_var == 0] = 1 # A dirty hack to avoid log(0)
        res = -self.sigma_f**2 * loc_var_gamma * np.log(loc_var) * np.exp(-loc_var_gamma)
        return res

    def get_derivative_function_list(self, params):
        ge = GammaExponential(params)
        return [ge._dge_dsigmaf, ge._dge_dl, ge._dge_dgamma]


# class ScaledSquaredExponential(CovarianceFamily):
#     """A class, representing the squared-exponential covariance functions family."""
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
#         return (1e-2, None), (1+1e-2, None), (1e-5, None)
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
#         return np.exp(-r**2 / (2*(np.log(l)**2))) * sigma_f**2 + gaussian_noise_term(sigma_l, x, y)
#
#     def _dse_dl(self, x, y):
#         r = np.linalg.norm(x - y, axis=0)
#         return (np.exp(-r**2 / (2*(np.log(self.l)**2))) * self.sigma_f**2) * (r**2 / (self.l * (np.log(self.l)**3)))
#
#     def _dse_dsigmaf(self, x, y):
#         r = np.linalg.norm(x - y, axis=0)
#         return 2 * self.sigma_f * np.exp(-r**2 / (2*(np.log(self.l)**2)))
#
#     def get_derivative_function_list(self, params):
#         sse = ScaledSquaredExponential(params)
#         return [sse._dse_dsigmaf, sse._dse_dl]

# def matern_cov(sigma, nu, l):
#     """Matern covariance function"""
#     def f(x, y):
#         r = np.linalg.norm(x - y)
#         if r == 0:
#             return 1.0
#         # print(r)
#         anc_var = np.sqrt(2.0 * nu) * r / l
#         # print(anc_var)
#         return sigma**2 *(2.0 ** (1.0 - nu) / gamma(nu)) * (anc_var ** nu) * kv(nu, anc_var)
#     return f


class Matern(CovarianceFamily):
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

    def covariance_function(self, x, y, w=np.NaN):
        if np.all(np.isnan(w)):
            l = self.l
            nu = self.nu
            sigma_f = self.sigma_f
            sigma_l = self.sigma_l
        else:
            sigma_f = w[0]
            l = w[1]
            nu = w[2]
            sigma_l = w[3]
        r = np.linalg.norm(x - y, axis=0)
        anc_var = np.sqrt(2.0 * nu) * r / l
        res = sigma_f**2 *(2.0 ** (1.0 - nu) / gamma(nu)) * (anc_var ** nu) * kv(nu, anc_var)
        res[r == 0] = sigma_f**2
        res += gaussian_noise_term(sigma_l, x, y)
        return res

    @staticmethod
    def get_bounds():
        return (1e-2, None), (1e-2, None), (1e-2, None), (1e-5, None)

    def _dm_dl(self, x, y):
        return 1e8 * (self.covariance_function(x, y, w=(self.get_params() + np.array([0, 1e-8, 0, 0]))) -
                      self.covariance_function(x, y))

    def _dm_dnu(self, x, y):
        return 1e8 * (self.covariance_function(x, y, w=(self.get_params() + np.array([0, 0, 1e-8, 0]))) -
                      self.covariance_function(x, y))

    def _dm_dsigmaf(self, x, y):
        r = np.linalg.norm(x - y, axis=0)
        anc_var = np.sqrt(2.0 * self.nu) * r / self.l
        res = 2 * self.sigma_f * (2.0 ** (1.0 - self.nu) / gamma(self.nu)) * (anc_var ** self.nu) * kv(self.nu,
                                                                                                        anc_var)
        res[r == 0] = 2 * self.sigma_f
        return res

    def get_derivative_function_list(self, params):
        m = аn(params)
        return [m._dm_dsigmaf, m._dm_dl, m._dm_dnu]


class ExpScaledSquaredExponential(CovarianceFamily):
    """A class, representing the squared-exponential covariance functions family
    with hyper-parameters in exponential scale."""

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
        return (-2, 10), (-2, 10), (-5, 10)

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
        return np.exp(-r**2 / (2*(np.exp(2 * l)))) * np.exp(sigma_f * 2) + gaussian_noise_term(np.exp(sigma_l), x, y)

    def _dcov_dl(self, x, y):
        r = np.linalg.norm(x - y, axis=0)
        return (np.exp(-r**2 / (2*(np.exp(self.l*2)))) * np.exp(self.sigma_f*2)) * (r**2 / (np.exp(self.l * 3))) * np.exp(self.l)

    def _dcov_dsigmaf(self, x, y):
        r = np.linalg.norm(x - y, axis=0)
        return np.exp(-r**2 / (2*(np.exp(2 * self.l)))) * 2 * np.exp(self.sigma_f * 2)

    def get_noise_derivative(self, points_num):
        """
        :return: The coefficient for the identity matrix in the derivative of the covariance w.r.t. noise variance.
        """
        return 2 * np.exp(2 * self.sigma_l) * np.eye(points_num)

    def get_derivative_function_list(self, params):
        cov_obj = ExpScaledSquaredExponential(params)
        return [cov_obj._dcov_dsigmaf, cov_obj._dcov_dl]


# if __name__ == '__main__':
#     alpha = 3
#     l = 0.2
#     sigma = 2
#     f = matern_cov(sigma, alpha, l)
#     x = np.array([[1], [1]])
#     y = np.array([[0], [0]])
#     print(np.linalg.norm(x - y))
#     print(f(x, y))