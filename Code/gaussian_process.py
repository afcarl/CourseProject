import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as op
import copy

from covariance_functions import CovarianceFamily, covariance_mat, sigmoid, SquaredExponential


def gp_plot_reg_data(x, y, color):
    """
    :param x: points array
    :param y: target values array
    :param color: color
    :return: fugure
    """
    if (not isinstance(x, np.ndarray) or
            not isinstance(y, np.ndarray)):
        raise TypeError("The first two arguments must be numpy arrays")

    loc_x = x.reshape((x.size, ))
    loc_y = y.reshape((y.size, ))
    plt.plot(loc_x, loc_y, color)


def gp_plot_class_data(x, y, color1, color2):
    """
    :param x: points array
    :param y: target values array
    :param color1: color for the first class
    :param color2: color for the second class
    :return: figure
    """
    loc_y = y.reshape((y.shape[0],))
    plt.plot(x[0, loc_y == 1], x[1, loc_y == 1], color1)
    plt.plot(x[0, loc_y == -1], x[1, loc_y == -1], color2)


class GaussianProcess:

    def __init__(self, cov_obj, mean_function, class_reg):
        """
        :param cov_obj: object of the class CovarianceFamily
        :param mean_function: function, mean of the gaussian process
        :return: GaussianProcess object
        """
        if not isinstance(cov_obj, CovarianceFamily):
            raise TypeError("The covariance object is of the wrong type")
        if not hasattr(mean_function, '__call__'):
            raise TypeError("mean_function must be callable")
        if not (class_reg == "class" or class_reg == "reg"):
            raise ValueError("The last operand should be 'class' or 'reg'")

        self.covariance_fun = cov_obj.covariance_function
        self.covariance_obj = cov_obj
        self.mean_fun = mean_function
        self.type = class_reg

    def generate_data(self, dim, num, seed=None):
        """
        :param dim: dimensions of the generated data
        :param num: number of generated data pointts
        :param class_reg: regression or classification
        :return: tuple (data_points, labels or target values)
        """
        if dim <= 0:
            raise ValueError("dim must be greater then 0")
        if num <= 0:
            raise ValueError("num must be greater then 0")

        if not (seed is None):
            np.random.seed(seed)
        points = np.random.rand(dim, num)
        if dim == 1:
            points = np.sort(points)
        targets = self.sample(self.mean_fun, self.covariance_fun, points, seed)
        if self.type == "class":
            targets = sigmoid(targets)
            targets = np.sign(targets - np.ones(targets.shape) * 0.5)
        return points, targets.reshape(targets.size, 1)

    @staticmethod
    def sample(mean_func, cov_func, points, seed=None):
        """
        :param mean_func: mean function
        :param cov_func: covariance function
        :param points: data points
        :return: sample gaussian process values at given points
        """
        if not hasattr(cov_func, '__call__'):
            raise TypeError("cov_func must be callable")
        if not hasattr(mean_func, '__call__'):
            raise TypeError("mean_func must be callable")
        if not isinstance(points, np.ndarray):
            raise TypeError("points must be a numpy array")

        cov_mat = covariance_mat(cov_func, points, points)
        m_v = np.array([mean_func(point) for point in points.T.tolist()])
        mean_vector = m_v.reshape((m_v.size,))
        if not (seed is None):
            np.random.seed(seed)
        res = np.random.multivariate_normal(mean_vector, cov_mat)
        return res

    @staticmethod
    def sample_for_matrices(mean_vec, cov_mat):
        """
        :param mean_vec: mean vector
        :param cov_mat: cavariance matrix
        :return: sample gaussian process values at given points
        """
        if not (isinstance(mean_vec, np.ndarray) and
                isinstance(cov_mat, np.ndarray)):
            raise TypeError("points must be a numpy array")
        y = np.random.multivariate_normal(mean_vec.reshape((mean_vec.size,)), cov_mat)
        upper_bound = mean_vec + 3 * np.sqrt(np.diagonal(cov_mat).reshape(mean_vec.shape))
        lower_bound = mean_vec - 3 * np.sqrt(np.diagonal(cov_mat).reshape(mean_vec.shape))
        return mean_vec, upper_bound, lower_bound

    @staticmethod
    def _reg_get_ml(targets, cov_inv, cov_l):
        """
        :param targets: target_values list
        :param points_inv: inversed covariance at data points
        :return: marginal likelihood
        """
        n = targets.size
        return -((targets.T.dot(cov_inv)).dot(targets) + 2 * np.sum(np.log(np.diag(cov_l))) +
                 n * np.log(2 * math.pi))/2

    @staticmethod
    def _reg_get_ml_partial_derivative(targets, cov_inv, dk_dtheta_mat):
        """
        :param y: target value vector
        :param k_inv: inverse covariance matrix at data points
        :param dk_dtheta_mat: the matrix of partial derivatives of the covariance function at data points
        with respect to hyper-parameter theta
        :return: marginal likelihood partial derivative with respect to hyper-parameter theta
        """
        return ((((targets.T.dot(cov_inv)).dot(dk_dtheta_mat)).dot(cov_inv)).dot(targets) -
                np.trace(cov_inv.dot(dk_dtheta_mat))) / 2

    def _reg_get_ml_grad(self, points, targets, cov_inv, params):
        """
        :param targets: target values' vector
        :param cov_inv: inverse covariance matrix
        :param points: data points array
        :return: marginal likelihood gradient with respect to hyperparameters
        """
        derivative_matrix_list = self.covariance_obj.get_derivative_function_list(params)
        noise_derivative = 2 * params[-1] * np.eye(points.shape[1])
        return np.array([self._reg_get_ml_partial_derivative(targets, cov_inv, covariance_mat(func, points, points))
                         for func in derivative_matrix_list] +
                        [self._reg_get_ml_partial_derivative(targets, cov_inv, noise_derivative)])

    def _reg_oracle(self, points, targets, params):
        """
        :param points: data points array
        :param targets: target values vector
        :param params: hyper-parameters vector
        """
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        points_cov = covariance_mat(cov_fun, points, points)
        points_l = np.linalg.cholesky(points_cov)
        points_l_inv = np.linalg.inv(points_l)
        points_cov_inv = points_l_inv.T.dot(points_l_inv)
        marginal_likelihood = self._reg_get_ml(targets, points_cov_inv, points_l)[0, 0]
        gradient = self._reg_get_ml_grad(points, targets, points_cov_inv, params)
        gradient = gradient.reshape(gradient.size, )
        return marginal_likelihood, gradient

    def oracle(self, points, targets, params=None):
        """
        :param points: data points array
        :param targets: target values vector
        :param params: hyper-parameters vector
        """
        if params is None:
            params = self.covariance_obj.get_params()
        if self.type == "reg":
            return self._reg_oracle(points, targets, params)
        return 0

    def find_hyper_parameters(self, data_points, target_values):
        """
        Optimizes covariance hyper-parameters
        :param data_points: an array of data points
        :param target_values: target values' vector
        :return:
        """
        if not(isinstance(data_points, np.ndarray) and
               isinstance(target_values, np.ndarray)):
            raise TypeError("The operands must be of type numpy array")

        def loc_fun(w):
            loss, grad = self.oracle(data_points, target_values, w)
            return -loss

        def loc_grad(w):
            loss, grad = self.oracle(data_points, target_values, w)
            return -grad

        # print(self.covariance_obj.get_params())
        # print(loc_grad(self.covariance_obj.get_params()))
        # print(loc_fun(self.covariance_obj.get_params()))
        # exit()
        bnds = ((1e-2, None), (1e-2, None), (1e-5, None))
        res = op.minimize(loc_fun, self.covariance_obj.get_params(), args=(), method='L-BFGS-B', jac=loc_grad, bounds=bnds,
            options={'gtol': 1e-5, 'disp': False})
        optimal_params = res.x
        self.covariance_obj.set_params(optimal_params)

    def predict(self, test_points, training_points, training_targets):
        """
        :param test_points: array of test points
        :param training_points: training set array
        :param training_targets: target values at training points
        :return: array of target_values
        """
        k_x = covariance_mat(self.covariance_fun, training_points, training_points)
        k_x_inv = np.linalg.inv(k_x)

        k_x_test = covariance_mat(self.covariance_fun, training_points, test_points)
        k_test_x = covariance_mat(self.covariance_fun, test_points, training_points)
        k_test = covariance_mat(self.covariance_fun, test_points, test_points)

        new_mean = np.dot(np.dot(k_test_x, k_x_inv), training_targets)
        new_cov = k_test - np.dot(np.dot(k_test_x, k_x_inv), k_x_test)

        test_targets, up, low = self.sample_for_matrices(new_mean, new_cov)
        gp_plot_reg_data(test_points, up, 'r')
        gp_plot_reg_data(test_points, low, 'g')
        gp_plot_reg_data(test_points, test_targets, 'b')
        return test_targets

if __name__ == "__main__":
    data_params = np.array([1.0, 0.2, 0.1])
    data_covariance_obj = SquaredExponential(data_params)
    gp = GaussianProcess(data_covariance_obj, lambda x: 0, 'reg')
    num = 1000
    test_num = 2000
    dim = 1
    x_tr, y_tr = gp.generate_data(dim, num, seed=21)
    x_test, y_test = gp.generate_data(dim, test_num, seed = 10)
    model_params = np.array([2.0, 0.8, 0.4])
    model_covariance_obj = SquaredExponential(model_params)
    new_gp = GaussianProcess(model_covariance_obj, lambda x: 0, 'reg')
    new_gp.find_hyper_parameters(x_tr, y_tr)
    print(new_gp.covariance_obj.get_params())
    predicted_y_test = new_gp.predict(x_test, x_tr, y_tr)
    print(np.linalg.norm(y_test - predicted_y_test))
    gp_plot_reg_data(x_tr, y_tr, 'bx')
    plt.show()
