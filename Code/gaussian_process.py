import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as op
import copy
import scipy as sp

from covariance_functions import CovarianceFamily, covariance_mat, sigmoid


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
        :return: tuple (data_points, labels or target values)
        """
        if dim <= 0:
            raise ValueError("dim must be greater then 0")
        if num <= 0:
            raise ValueError("num must be greater then 0")

        if not (seed is None):
            np.random.seed(seed)
        points = np.random.rand(dim, num)
        targets = self.sample(self.mean_fun, self.covariance_fun, points, seed)
        if self.type == "class":
            targets = np.sign(targets)
        return points, targets.reshape((targets.size, 1))

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
        # y = np.random.multivariate_normal(mean_vec.reshape((mean_vec.size,)), cov_mat)
        upper_bound = mean_vec + 3 * np.sqrt(np.diagonal(cov_mat).reshape(mean_vec.shape))
        lower_bound = mean_vec - 3 * np.sqrt(np.diagonal(cov_mat).reshape(mean_vec.shape))
        return mean_vec, upper_bound, lower_bound

    @staticmethod
    def _reg_get_ml(targets, cov_inv, cov_l):
        """
        :param targets: target_values list
        :param cov_inv: inversed covariance at data points
        :return: marginal likelihood
        """
        n = targets.size
        return -((targets.T.dot(cov_inv)).dot(targets) + 2 * np.sum(np.log(np.diag(cov_l))) +
                 n * np.log(2 * math.pi))/2

    @staticmethod
    def _reg_get_ml_partial_derivative(targets, cov_inv, dk_dtheta_mat):
        """
        :param targets: target value vector
        :param cov_inv: inverse covariance matrix at data points
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
        marginal_likelihood = self._reg_get_ml(targets, points_cov_inv, points_l)
        gradient = self._reg_get_ml_grad(points, targets, points_cov_inv, params)
        gradient = gradient.reshape(gradient.size, )
        return marginal_likelihood, gradient

    def _reg_find_hyper_parameters(self, data_points, target_values):
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
            loss, grad = self._reg_oracle(data_points, target_values, w)
            return -loss

        def loc_grad(w):
            loss, grad = self._reg_oracle(data_points, target_values, w)
            return -grad

        bnds = ((1e-2, None), (1e-2, None), (1e-5, None))
        res = op.minimize(loc_fun, self.covariance_obj.get_params(), args=(), method='L-BFGS-B', jac=loc_grad,
                          bounds=bnds, options={'gtol': 1e-5, 'disp': True})
        optimal_params = res.x
        self.covariance_obj.set_params(optimal_params)

    def _reg_predict(self, test_points, training_points, training_targets):
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

    @staticmethod
    def _class_get_ml(f_opt, b, cov_inv):
        """
        :param f_opt: posterior mode
        :param b: I + W^(1/2) K W^(1/2), where W is minus hessian of the log likelihood at f_opt
        :param cov_inv: inverse covariance matrix at data points
        :return: the Laplace estimation of log marginal likelihood
        """
        l_b = np.linalg.cholesky(b)
        return -((f_opt.T.dot(cov_inv)).dot(f_opt) + 2 * np.sum(np.log(np.diag(l_b))))/2

    @staticmethod
    def _get_b(points_cov, hess_opt):
        """
        :param points_cov: covariance matrix
        :param hess_opt: -\nabla\nabla log p(y|f)|_{f_opt}
        :return: I + W^(1/2) K W^(1/2), where W is minus hessian of the log likelihood at f_opt
        """
        w = sp.linalg.sqrtm(hess_opt)
        b = (w.dot(points_cov)).dot(w)
        return np.eye(b.shape[0]) + b

    @staticmethod
    def _get_laplace_approximation(labels, cov_inv, cov_l):
        """
        :param labels: label vector
        :param cov_inv: inverse covariance matrix at data points
        :return: tuple (posterior mode f_opt, unnormalized posterior hessian at f_opt)
        """
        def loss(f):
            """log p(y|f)"""
            f = f.reshape((f.size, 1))
            return (np.sum(np.log(np.exp(-labels * f) + np.ones(labels.shape))) + (f.T.dot(cov_inv)).dot(f)/2 +
                    2 * np.sum(np.log(np.diag(cov_l)))/2)

        def hessian(f):
            """Hessian of the log p(y|f)"""
            f = f.reshape((f.size, 1))
            diag_vec = (-np.exp(f) / np.square(np.ones(f.shape) + np.exp(f)))
            return -np.diag(diag_vec.reshape((diag_vec.size, ))) + cov_inv

        def grad(f):
            f = f.reshape((f.size, 1))
            return (-((labels + np.ones(labels.shape))/2 - sigmoid(f)) + cov_inv.dot(f)).reshape((f_0.size,))

        f_0 = np.zeros(labels.shape)
        f_0 = f_0.reshape((f_0.size,))
        f_res = op.minimize(loss, f_0, args=(), method='L-BFGS-B', jac=grad,
                            options={'gtol': 1e-5, 'disp': False, 'maxiter': 1000})
        f_opt = f_res['x']
        return f_opt, hessian(f_opt) - cov_inv

    @staticmethod
    def _class_get_ml_partial_derivative(f_opt, k_inv, ancillary_mat, dk_dtheta_mat):
        """
        :param f_opt: posterior mode
        :param k_inv: inverse covariance matrix at data points
        :param ancillary_mat: (W^(-1) + K)^(-1), where W is hess_opt
        :param dk_dtheta_mat: the matrix of partial derivatives of the covariance function at data points
        with respect to hyper-parameter theta
        :return: marginal likelihood partial derivative with respect to hyper-parameter theta
        """
        return ((((f_opt.T.dot(k_inv)).dot(dk_dtheta_mat)).dot(k_inv)).dot(f_opt) -
                np.trace(ancillary_mat.dot(dk_dtheta_mat)))/2

    def _class_get_ml_grad(self, points, cov_inv, f_opt, anc_mat, params):
        """
        :param points: data points array
        :param cov_inv: inverse covariance matrix
        :param params: hyper-parameters vector
        :param f_opt: posterior mode
        :param anc_mat: ancillary matrix
        :return: marginal likelihood gradient with respect to hyper-parameters
        """
        derivative_matrix_list = self.covariance_obj.get_derivative_function_list(params)
        noise_derivative = 2 * params[-1] * np.eye(points.shape[1])
        return np.array([self._class_get_ml_partial_derivative(f_opt, cov_inv, anc_mat,
                                                               covariance_mat(func, points, points))
                         for func in derivative_matrix_list] +
                        [self._class_get_ml_partial_derivative(f_opt, cov_inv, anc_mat,  noise_derivative)])

    def _class_oracle(self, points, f_opt, hess_opt, params):
        """
        :param points: data points array
        :param f_opt: posterior mode
        :param hess_opt: some weird hessian (-\nabla \nabla \log p(y|f_opt))
        :param params: hyper-parameters vector
        """
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        points_cov = covariance_mat(cov_fun, points, points)
        points_l = np.linalg.cholesky(points_cov)
        points_l_inv = np.linalg.inv(points_l)
        points_cov_inv = points_l_inv.T.dot(points_l_inv)

        b_mat = self._get_b(points_cov, hess_opt)
        marginal_likelihood = self._class_get_ml(f_opt, b_mat, points_cov_inv)

        anc_mat = np.linalg.inv(np.linalg.inv(hess_opt) + points_cov)
        gradient = self._class_get_ml_grad(points, points_cov_inv, f_opt, anc_mat, params)

        return marginal_likelihood, gradient

    def _class_find_hyper_parameters(self, points, labels, max_iter=10):
        """
        :param points: data points
        :param labels: class labels at data points
        :param max_iter: maximim number of iterations
        :return: optimal hyper-parameters
        """
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_fun = cov_obj.covariance_function
        bnds = ((1e-2, None), (1e-2, None), (1e-5, None))
        w0 = self.covariance_obj.get_params()

        def func(w):
            loss, _ = self._class_oracle(points, f_opt, hess_opt, w)
            return -loss

        def grad(w):
            _, gradient = self._class_oracle(points, f_opt, hess_opt, w)
            return -gradient

        for i in range(max_iter):
            points_cov = covariance_mat(cov_fun, points, points)
            points_l = np.linalg.cholesky(points_cov)
            points_l_inv = np.linalg.inv(points_l)
            points_cov_inv = points_l_inv.T.dot(points_l_inv)
            det_k = 2 * np.sum(np.log(np.diag(points_l)))

            f_opt, hess_opt = self._get_laplace_approximation(labels, points_cov_inv, points_l)
            w_res = op.minimize(func, w0, args=(), method='L-BFGS-B', jac=grad, bounds=bnds,
                                options={'ftol': 1e-5, 'disp': False, 'maxiter': 1})
            w0 = w_res['x']
            if not(i%10):
                print("Iteration ", i)
                print("Hyper-parameters at iteration ", i, ": ", w0)
            cov_obj.set_params(w0)
        self.covariance_obj = copy.deepcopy(cov_obj)

    def _class_predict(self, test_points, training_points, training_labels):
        """
        :param test_points: test data points
        :param training_points: training data points
        :param training_labels: class labels at training points
        :return: prediction of class labels at given test points
        """
        cov_fun = self.covariance_obj.covariance_function
        points_cov = covariance_mat(cov_fun, training_points, training_points)
        points_l = np.linalg.cholesky(points_cov)
        points_l_inv = np.linalg.inv(points_l)
        points_cov_inv = points_l_inv.T.dot(points_l_inv)
        f_opt, hess_opt = self._get_laplace_approximation(training_labels, points_cov_inv, points_l)
        k_test_x = covariance_mat(cov_fun, test_points, training_points)

        f_test = np.dot(np.dot(k_test_x, points_cov_inv), f_opt)
        loc_y_test = np.sign(f_test.reshape((f_test.size, 1)))

        return loc_y_test

    @staticmethod
    def _class_get_implicit_ml_partial_derivative(f_opt, k_inv, ancillary_mat, dk_dtheta_mat, labels):
        """
        :param f_opt: posterior mode
        :param k_inv: inverse covariance matrix at data points
        :param ancillary_mat: (W^(-1) + K)^(-1), where W is hess_opt
        :param dk_dtheta_mat: matrix of partial derivatives of covariance matrix wrt theta
        :param labels: labels at data points
        :return: implicit part of the partial derivative of ml wrt theta.
        """
        # First we will compute the derivative of q (approx. ml) wrt f_opt.
        anc_diag = np.diag(ancillary_mat)
        f_exp = np.exp(f_opt)
        dw_df = f_exp * (f_exp - 1) / (f_exp + 1)**3
        dq_df = -(anc_diag * (dw_df)) / 2

        # Now we compute the derivative of f_opt wrt theta_j

        df_opt_dtheta = ((ancillary_mat.dot(k_inv)).dot(dk_dtheta_mat)).dot(
            labels.reshape(labels.size,) / (1 + np.exp(labels.reshape(labels.size,) * f_opt)))

        # Now we combine everything to get the derivative
        return np.dot(dq_df, df_opt_dtheta)

    def _class_get_ml_full_grad(self, points, labels, cov_inv, f_opt, anc_mat, params):
        """
        :param points: data points array
        :param labels: labels at training points
        :param cov_inv: inverse covariance matrix
        :param params: hyper-parameters vector
        :param f_opt: posterior mode
        :param anc_mat: ancillary matrix
        :return: marginal likelihood gradient with respect to hyper-parameters
        """
        derivative_matrix_list = self.covariance_obj.get_derivative_function_list(params)
        noise_derivative = 2 * params[-1] * np.eye(points.shape[1])
        return np.array([self._class_get_implicit_ml_partial_derivative(f_opt, cov_inv, anc_mat,
                                                               covariance_mat(func, points, points), labels) +
                         self._class_get_ml_partial_derivative(f_opt, cov_inv, anc_mat,
                                                               covariance_mat(func, points, points))
                         for func in derivative_matrix_list] +
                        [self._class_get_implicit_ml_partial_derivative(f_opt, cov_inv, anc_mat,  noise_derivative,
                                                                        labels) +
                         self._class_get_ml_partial_derivative(f_opt, cov_inv, anc_mat,  noise_derivative)])

    @staticmethod
    def _class_get_full_ml(f_opt, b, cov_inv, labels):
        """
        :param f_opt: posterior mode
        :param b: I + W^(1/2) K W^(1/2), where W is minus hessian of the log likelihood at f_opt
        :param cov_inv: inverse covariance matrix at data points
        :return: the Laplace estimation of log marginal likelihood
        """
        l_b = np.linalg.cholesky(b)
        return -(((f_opt.T.dot(cov_inv)).dot(f_opt) + 2 * np.sum(np.log(np.diag(l_b))))/2 +
                 np.sum(np.log(1 + np.exp(-labels * f_opt))))

    def _class_alternative_oracle(self, points, labels, f_opt, hess_opt, params):
        """
        :param points: data points array
        :param labels: labels at data points
        :param f_opt: posterior mode
        :param hess_opt: some weird hessian (-\nabla \nabla \log p(y|f_opt))
        :param params: hyper-parameters vector
        """
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        points_cov = covariance_mat(cov_fun, points, points)
        points_l = np.linalg.cholesky(points_cov)
        points_l_inv = np.linalg.inv(points_l)
        points_cov_inv = points_l_inv.T.dot(points_l_inv)

        b_mat = self._get_b(points_cov, hess_opt)
        marginal_likelihood = self._class_get_full_ml(f_opt, b_mat, points_cov_inv, labels.reshape(labels.size, ))

        anc_mat = np.linalg.inv(np.linalg.inv(hess_opt) + points_cov)
        gradient = self._class_get_ml_full_grad(points, labels.reshape(labels.size, ), points_cov_inv,
                                                f_opt, anc_mat, params)

        return marginal_likelihood, gradient

    def _class_alternative_find_hyper_parameters(self, points, labels, max_iter=10):
        """
        :param points: data points
        :param labels: class labels at data points
        :param max_iter: maximim number of iterations
        :return: optimal hyper-parameters
        """
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_fun = cov_obj.covariance_function
        bnds = ((1e-2, None), (1e-2, None), (1e-5, None))
        w0 = self.covariance_obj.get_params()

        def func(w):
            loss, _ = self._class_alternative_oracle(points, labels, f_opt, hess_opt, w)
            return -loss

        def grad(w):
            _, gradient = self._class_alternative_oracle(points, labels, f_opt, hess_opt, w)
            return -gradient

        for i in range(max_iter):
            points_cov = covariance_mat(cov_fun, points, points)
            points_l = np.linalg.cholesky(points_cov)
            points_l_inv = np.linalg.inv(points_l)
            points_cov_inv = points_l_inv.T.dot(points_l_inv)
            # det_k = 2 * np.sum(np.log(np.diag(points_l)))

            f_opt, hess_opt = self._get_laplace_approximation(labels, points_cov_inv, points_l)
            w_res = op.minimize(func, w0, args=(), method='L-BFGS-B', jac=grad, bounds=bnds,
                                options={'ftol': 1e-5, 'disp': False, 'maxiter': 1})
            w0 = w_res['x']
            if not(i%10):
                print("Iteration ", i, ": ", func(w0), np.linalg.norm(grad(w0)))
                print("Hyper-parameters at iteration ", i, ": ", w0)
            cov_obj.set_params(w0)
        self.covariance_obj = copy.deepcopy(cov_obj)