import numpy as np
import math
import scipy.optimize as op
import copy
import scipy as sp
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from covariance_functions import CovarianceFamily, sigmoid


def minimize_wrapper(func, x0, mydisp=False, **kwargs):

    start = time.time()
    aux = {'start': time.time(), 'total': 0., 'it': 0}
    def callback(w):
        # print(start)
        # print(total_time)
        # total_time
        aux['total'] += time.time() - aux['start']
        if mydisp:
            print("Hyper-parameters at iteration", aux['it'], ":", w)
        fun, _ = func(w)
        fun_list.append(fun)
        # total_time += time.time() - start
        time_list.append(aux['total'])
        w_list.append(w)
        aux['it'] += 1
        aux['start'] = time.time()
    # print(start)
    w_list = []
    time_list = []
    fun_list = []
    callback(x0)

    out = op.minimize(func, x0, jac=True, callback=callback, **kwargs)

    return out, w_list, time_list, fun_list


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

    def generate_data(self, tr_points, test_points, seed=None):
        """
        :param dim: dimensions of the generated data
        :param tr_points: training data points
        :param test_points: testing data points
        :return: tuple (training data points, training labels or target values, test data points, test labels or target
        values)
        """

        if not (seed is None):
            np.random.seed(seed)
        targets = self.sample(self.mean_fun, self.covariance_fun, np.hstack((tr_points, test_points)), seed)
        if self.type == "class":
            targets = np.sign(targets)
        targets = targets.reshape((targets.size, 1))
        return targets[:tr_points.shape[1], :], targets[tr_points.shape[1]:, :]

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

        cov_mat = cov_func(points, points)
        m_v = np.array([mean_func(point) for point in points.T.tolist()])
        mean_vector = m_v.reshape((m_v.size,))
        if not (seed is None):
            np.random.seed(seed)
        print(cov_mat.shape)
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
        # print(np.diagonal(cov_mat).reshape(mean_vec.shape))
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
        # noise_derivative = 2 * params[-1] * np.eye(points.shape[1])
        noise_derivative = self.covariance_obj.get_noise_derivative(points.shape[1])
        return np.array([self._reg_get_ml_partial_derivative(targets, cov_inv, func(points, points))
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
        points_cov = cov_fun(points, points)
        points_l = np.linalg.cholesky(points_cov)
        points_l_inv = np.linalg.inv(points_l)
        points_cov_inv = points_l_inv.T.dot(points_l_inv)
        marginal_likelihood = self._reg_get_ml(targets, points_cov_inv, points_l)
        gradient = self._reg_get_ml_grad(points, targets, points_cov_inv, params)
        gradient = gradient.reshape(gradient.size, )
        return marginal_likelihood, gradient

    def _reg_find_hyper_parameters(self, data_points, target_values, max_iter=None):
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
            return -loss, -grad

        # w0 = self.covariance_obj.get_params()
        # f1, g1 = loc_fun(w0 + np.array([0, 0, 1e-8]))
        # f2, g2 = loc_fun(w0)
        # print(g1)
        # print((f1 - f2) * 1e8)
        # exit(0)
        bnds = self.covariance_obj.get_bounds()
        if max_iter is None:
            max_iter = np.inf
        res, w_list, time_list, fun_lst = minimize_wrapper(loc_fun, self.covariance_obj.get_params(), method='L-BFGS-B',
                                                  mydisp=True, bounds=bnds, options={'gtol': 1e-8, 'ftol': 0,
                                                                                     'maxiter': max_iter})
        optimal_params = res.x
        self.covariance_obj.set_params(optimal_params)
        return w_list, time_list, fun_lst

    def _reg_predict(self, test_points, training_points, training_targets):
        """
        :param test_points: array of test points
        :param training_points: training set array
        :param training_targets: target values at training points
        :return: array of target_values
        """
        k_x = self.covariance_fun(training_points, training_points)
        k_x_inv = np.linalg.inv(k_x)

        k_test_x = self.covariance_fun(test_points, training_points)
        k_test = self.covariance_fun(test_points, test_points)

        new_mean = np.dot(np.dot(k_test_x, k_x_inv), training_targets)
        new_cov = k_test - np.dot(np.dot(k_test_x, k_x_inv), k_test_x.T)

        test_targets, up, low = self.sample_for_matrices(new_mean, new_cov)
        # if test_points.shape[0] == 1:
        #     gp_plot_reg_data(test_points, up, 'r')
        #     gp_plot_reg_data(test_points, low, 'g')
        #     gp_plot_reg_data(test_points, test_targets, 'b')
        return test_targets, low, up

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
    def _get_laplace_approximation(labels, cov_inv, cov_l, max_iter=1000):
        """
        :param labels: label vector
        :param cov_inv: inverse covariance matrix at data points
        :param cov_l: cholesky decomposition matrix of the covariance matrix
        :param max_iter: maximum number of iterations for the optimization
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
                            options={'gtol': 1e-5, 'disp': False, 'maxiter': max_iter})
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
        !! If the covariance function does not provide a derivative w.r.t. to the noise variance (the last parameter of
        the covariance function), it is assumed to be equal to 2 * noise variance * I. Else it is assumed to be
        derivative_matrix_list[-1] * I.
        :param points: data points array
        :param cov_inv: inverse covariance matrix
        :param params: hyper-parameters vector
        :param f_opt: posterior mode
        :param anc_mat: ancillary matrix
        :return: marginal likelihood gradient with respect to hyper-parameters
        """
        derivative_matrix_list = self.covariance_obj.get_derivative_function_list(params)
        noise_derivative = self.covariance_obj.get_noise_derivative(points.shape[1])
        # print(noise_derivative.shape)
        return np.array([self._class_get_ml_partial_derivative(f_opt, cov_inv, anc_mat,
                                                               func(points, points))
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
        points_cov = cov_fun(points, points)
        points_l = np.linalg.cholesky(points_cov)
        points_l_inv = np.linalg.inv(points_l)
        points_cov_inv = points_l_inv.T.dot(points_l_inv)

        b_mat = self._get_b(points_cov, hess_opt)
        marginal_likelihood = self._class_get_ml(f_opt, b_mat, points_cov_inv)

        anc_mat = np.linalg.inv(np.linalg.inv(hess_opt) + points_cov)
        gradient = self._class_get_ml_grad(points, points_cov_inv, f_opt, anc_mat, params)

        return marginal_likelihood, gradient

    def _class_find_hyper_parameters(self, points, labels, max_iter=10, alternate=False):
        """
        optimizes the self.covariance_obj hyper-parameters
        :param points: data points
        :param labels: class labels at data points
        :param max_iter: maximim number of iterations
        :return: a list of hyper-parameters values at different iterations and a list of times iteration-wise
        """
        if alternate:
            return self._class_alternative_find_hyper_parameters(points, labels, max_iter)
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_fun = cov_obj.covariance_function
        bnds = self.covariance_obj.get_bounds()
        w0 = self.covariance_obj.get_params()
        w_list = []
        time_list = []

        def func(w):
            loss, grad = self._class_oracle(points, f_opt, hess_opt, w)
            return -loss, -grad

        # def grad(w):
        #     _, gradient = self._class_oracle(points, f_opt, hess_opt, w)
        #     return -gradient

        start = time.time()
        for i in range(max_iter):
            points_cov = cov_fun(points, points)
            points_l = np.linalg.cholesky(points_cov)
            points_l_inv = np.linalg.inv(points_l)
            points_cov_inv = points_l_inv.T.dot(points_l_inv)

            f_opt, hess_opt = self._get_laplace_approximation(labels, points_cov_inv, points_l, max_iter=20)

            # f1, g1 = func(w0)
            # f2, g2 = func(w0 + np.array([0, 0, 1e-8]).reshape(w0.shape))
            # print((f2 - f1) * 1e8)
            # print(g1)
            # print(g2)
            # exit(0)

            w_res = op.minimize(func, w0, args=(), method='L-BFGS-B', jac=True, bounds=bnds,
                                options={'ftol': 1e-5, 'disp': False, 'maxiter': 20})
            w0 = w_res['x']
            if not(i % 10):
                print("Iteration ", i)
                print("Hyper-parameters at iteration ", i, ": ", w0)

            w_list.append(w0)
            time_list.append(time.time() - start)
            cov_obj.set_params(w0)
        self.covariance_obj = copy.deepcopy(cov_obj)
        return w_list, time_list

    def _class_predict(self, test_points, training_points, training_labels):
        """
        :param test_points: test data points
        :param training_points: training data points
        :param training_labels: class labels at training points
        :return: prediction of class labels at given test points
        """
        cov_fun = self.covariance_obj.covariance_function
        points_cov = cov_fun(training_points, training_points)
        points_l = np.linalg.cholesky(points_cov)
        points_l_inv = np.linalg.inv(points_l)
        points_cov_inv = points_l_inv.T.dot(points_l_inv)
        f_opt, hess_opt = self._get_laplace_approximation(training_labels, points_cov_inv, points_l)
        k_test_x = cov_fun(test_points, training_points)

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
        dq_df = -(anc_diag * dw_df) / 2

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
        # noise_derivative = 2 * params[-1] * np.eye(points.shape[1])
        noise_derivative = self.covariance_obj.get_noise_derivative(points.shape[1])

        return np.array([self._class_get_implicit_ml_partial_derivative(f_opt, cov_inv, anc_mat,
                                                                        func(points, points), labels) +
                         self._class_get_ml_partial_derivative(f_opt, cov_inv, anc_mat,
                                                               func(points, points))
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
        points_cov = cov_fun(points, points)
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
        optimizes self.covariance_obj hyper-parameters
        :param points: data points
        :param labels: class labels at data points
        :param max_iter: maximim number of iterations
        :return: a list of hyper-parameters values at different iterations and a list of times iteration-wise
        """
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_fun = cov_obj.covariance_function
        w0 = self.covariance_obj.get_params()
        bnds = self.covariance_obj.get_bounds()
        w_list = []
        time_list = []

        def func(w):
            loss, _ = self._class_alternative_oracle(points, labels, f_opt, hess_opt, w)
            return -loss

        def grad(w):
            _, gradient = self._class_alternative_oracle(points, labels, f_opt, hess_opt, w)
            return -gradient

        start = time.time()

        for i in range(max_iter):
            points_cov = cov_fun(points, points)
            points_l = np.linalg.cholesky(points_cov)
            points_l_inv = np.linalg.inv(points_l)
            points_cov_inv = points_l_inv.T.dot(points_l_inv)
            # det_k = 2 * np.sum(np.log(np.diag(points_l)))

            f_opt, hess_opt = self._get_laplace_approximation(labels, points_cov_inv, points_l, max_iter=np.inf)
            w_res = op.minimize(func, w0, args=(), method='L-BFGS-B', jac=grad, bounds=bnds,
                                options={'ftol': 1e-5, 'disp': False, 'maxiter': 50})
            w0 = w_res['x']
            if not(i % 10):
                print("Iteration ", i, ": ", func(w0), np.linalg.norm(grad(w0)))
                print("Hyper-parameters at iteration ", i, ": ", w0)
            w_list.append(w0)
            time_list.append(time.time() - start)
            cov_obj.set_params(w0)
        self.covariance_obj = copy.deepcopy(cov_obj)
        return w_list, time_list

    def find_hyper_parameters(self, *args, **kwargs):
        if self.type == "class":
            return self._class_find_hyper_parameters(*args, **kwargs)
        elif self.type == "reg":
            return self._reg_find_hyper_parameters(*args, **kwargs)
        else:
            raise ValueError("GP type should be either 'class' or 'reg'")

    def predict(self, *args, **kwargs):
        if self.type == "class":
            return self._class_predict(*args, **kwargs)
        elif self.type == "reg":
            return self._reg_predict(*args, **kwargs)
        else:
            raise ValueError("GP type should be either 'class' or 'reg'")

    # def reg_plot_marginal_likelihood(self, points, targets):
    #     l_list = np.linspace(0.01, 10, 1000)
    #     l_list = np.log(l_list)
    #     fun_lst = []
    #     params = self.covariance_obj.get_params()
    #     it = 0
    #     for l in l_list:
    #         params[1] = l
    #         if it % 10:
    #             print(it)
    #         self.covariance_obj.set_params(params)
    #         cov_fun = self.covariance_obj.covariance_function
    #         points_cov = cov_fun(points, points)
    #         points_l = np.linalg.cholesky(points_cov)
    #         points_l_inv = np.linalg.inv(points_l)
    #         points_cov_inv = points_l_inv.T.dot(points_l_inv)
    #         fun_lst.append(self._reg_get_ml(targets, points_cov_inv, points_l).reshape(1))
    #         it += 1
    #     plt.plot(l_list, fun_lst)
    #     plt.show()

    def reg_find_inducing_inputs(self, data_points, target_values, num_inputs, max_iter=None):
        """
        Find inducing inputs for gp-regression using variational learning approach
        :param data_points: data points
        :param target_values: target values at data points
        :param num_inputs: number of inducing inputs to be found
        :param max_iter: maximum number of iterations
        :return:
        """
        if not(isinstance(data_points, np.ndarray) and
               isinstance(target_values, np.ndarray)):
            raise TypeError("The operands must be of type numpy array")

        dim = data_points.shape[0]
        param_len = self.covariance_obj.get_params().size

        def loc_fun(w):
            # print("Params:", w[:param_len])

            ind_points = (w[param_len:]).reshape((dim, num_inputs)) # has to be rewritten for multidimensional case
            loss, grad = self._reg_inducing_points_oracle(data_points, target_values, w[:param_len], ind_points)
            return -loss, -grad

        bnds = tuple(list(self.covariance_obj.get_bounds()) + [(1e-2, 1)] * num_inputs * dim)
        if max_iter is None:
            max_iter = np.inf
        inputs = data_points[:, :num_inputs] + np.random.normal(0, 0.1, (dim, num_inputs))
        np.random.seed(15)
        w0 = np.concatenate((self.covariance_obj.get_params(), inputs.ravel()))
        # f2, g2 = loc_fun(w0)
        # print("Gradient:", g2)
        # for i in range(w0.size):
        #     diff = np.zeros(w0.shape)
        #     diff[i] = 1e-8
        #     f1, g1 = loc_fun(w0 + diff)
        #     print("Difference:", (f1 - f2) * 1e8)
        # exit(0)
        res, w_list, time_list, fun_lst = minimize_wrapper(loc_fun, w0, method='L-BFGS-B',
                                                  mydisp=True, bounds=bnds, options={'gtol': 1e-8, 'ftol': 0,
                                                                                     'maxiter': max_iter})
        optimal_params = res.x[:-num_inputs*dim]
        # print(optimal_params)
        inducing_point = res.x[-num_inputs*dim:]
        inducing_point = inducing_point.reshape((dim, num_inputs))
        self.covariance_obj.set_params(optimal_params)

        cov_fun = self.covariance_obj.covariance_function
        sigma = optimal_params[-1]
        K_mm = cov_fun(inducing_point, inducing_point)
        K_mn = cov_fun(inducing_point, data_points)
        K_nm = K_mn.T
        Sigma = np.linalg.inv(K_mm + K_mn.dot(K_nm)/sigma**2)
        mu = sigma**(-2) * K_mm.dot(Sigma).dot(K_mn).dot(target_values)
        A = K_mm.dot(Sigma).dot(K_mm)
        # print(mu)
        return inducing_point, mu, A, w_list, time_list, fun_lst

    def _reg_inducing_points_oracle(self, points, targets, params, ind_points):
        """
        :param points: data points array
        :param targets: target values vector
        :param params: hyper-parameters vector
        :param ind_points: inducing points
        """
        sigma = params[-1]
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        K_mm = cov_fun(ind_points, ind_points)
        # print(pairwise_distance(ind_points, ind_points))
        # print(ind_points)
        # print(np.linalg.det(K_mm))
        K_mm_inv = np.linalg.inv(K_mm) # use cholesky?
        K_nm = cov_fun(points, ind_points)
        K_mn = K_nm.T
        Q_nn = (K_nm.dot(K_mm_inv)).dot(K_mn)
        B = Q_nn + np.eye(Q_nn.shape[0]) * sigma**2
        # print(params)
        B_l = np.linalg.cholesky(B)
        B_l_inv = np.linalg.inv(B_l)
        B_inv = B_l_inv.T.dot(B_l_inv)
        zero = np.array([[0]])
        K_nn_diag = cov_fun(zero, zero)
        F_v = - np.sum(np.log(np.diag(B_l))) - targets.T.dot(B_inv).dot(targets)/2 - \
              np.sum(K_nn_diag - np.diag(Q_nn)) / (2 * sigma**2)

        # Gradient
        gradient = []
        derivative_matrix_list = cov_obj.get_derivative_function_list(params)
        for func in derivative_matrix_list:
            dK_nm = func(points, ind_points)
            dK_mn = dK_nm.T
            dK_mm = func(ind_points, ind_points)
            dK_mm_inv = - K_mm_inv.dot(dK_mm.dot(K_mm_inv))
            dB_dtheta = (dK_nm.dot(K_mm_inv) + K_nm.dot(dK_mm_inv)).dot(K_mn) + K_nm.dot(K_mm_inv.dot(dK_mn))
            dK_nn = func(zero, zero)
            gradient.append(self._reg_get_lower_bound_partial_derivative(targets, dB_dtheta, dB_dtheta, B_inv, sigma,
                                                                         dK_nn))

        # sigma derivative
        dK_mm = cov_obj.get_noise_derivative(K_mm.shape[0])
        dK_mm_inv = - K_mm_inv.dot(dK_mm.dot(K_mm_inv))
        dQ_dtheta = K_nm.dot(dK_mm_inv).dot(K_mn)
        dB_dtheta = dQ_dtheta + 2 * sigma * np.eye(Q_nn.shape[0])
        # dK_nn = (zero[:, :, None], zero[:, None, :])
        dK_nn = 2 * sigma
        gradient.append(self._reg_get_lower_bound_partial_derivative(targets, dB_dtheta, dQ_dtheta, B_inv, sigma,
                                                                     dK_nn))
        gradient[-1] += np.sum(K_nn_diag - np.diag(Q_nn)) / sigma**3

        # inducing points derivatives
        K_mn_derivatives = cov_obj.covariance_derivative(ind_points, points)
        K_mm_derivatives = cov_obj.covariance_derivative(ind_points, ind_points)
        for j in range(ind_points.shape[0]):
            for i in range(ind_points.shape[1]):
                dK_mn = np.zeros(K_mn.shape)
                dK_mn[i, :] = K_mn_derivatives[j, i, :]
                dK_nm = dK_mn.T
                dK_mm = np.zeros(K_mm.shape)
                dK_mm[i, :] = K_mm_derivatives[j, i, :]
                dK_mm[:, i] = K_mm_derivatives[j, i, :].T
                dK_mm_inv = - K_mm_inv.dot(dK_mm.dot(K_mm_inv))
                dB_dtheta = (dK_nm.dot(K_mm_inv) + K_nm.dot(dK_mm_inv)).dot(K_mn) + K_nm.dot(K_mm_inv.dot(dK_mn))
                dK_nn = 0
                gradient.append(self._reg_get_lower_bound_partial_derivative(targets, dB_dtheta, dB_dtheta, B_inv, sigma,
                                                                         dK_nn))
        return F_v[0, 0], np.array(gradient)

    def _reg_get_lower_bound_partial_derivative(self, y, dB_dtheta_mat, dQ_dtheta_mat, B_inv, sigma, dK_nn_dtheta):
        """
        The derivative of the variational lower bound for evidence wrt to theta. Note that if theta is noise variance,
        the result is incorrect.
        :param dB_dtheta_mat: B derivative wrt theta
        :param dQ_dtheta_mat: Q_nn derivative wrt theta
        :param B_inv: inverse B matrix
        :param sigma: noise variance
        :param dK_nn_dtheta: K_nn diagonal elements derivative wrt theta
        :param y: target values at data points
        :return: partial derivative of the lower bound F_v wrt theta
        """
        return (-np.trace(B_inv.dot(dB_dtheta_mat)) / 2 + y.T.dot(B_inv.dot(dB_dtheta_mat.dot(B_inv.dot(y)))) / 2 -
                np.sum(dK_nn_dtheta - np.diag(dQ_dtheta_mat)) / (2 * sigma**2))[0, 0]

    def reg_inducing_points_predict(self, ind_points, expectation, covariance, test_points):
        """
        Predict new values given inducing points
        :param ind_points: inducing points
        :param expectation: expectation at inducing points
        :param covariance: covariance at inducing points
        :param test_points: test points
        :return: predicted values at inducing points
        """
        cov_fun = self.covariance_obj.covariance_function
        K_xm = cov_fun(test_points, ind_points)
        K_mx = K_xm.T
        K_mm = cov_fun(ind_points, ind_points)
        K_xx = cov_fun(test_points, test_points)
        K_mm_inv = np.linalg.inv(K_mm)

        new_mean = K_xm.dot(K_mm_inv).dot(expectation)
        new_cov = K_xx - K_xm.dot(K_mm_inv).dot(K_mx) + K_xm.dot(K_mm_inv).dot(covariance).dot(K_mm_inv).dot(K_mx)

        test_targets, up, low = self.sample_for_matrices(new_mean, new_cov)
        return test_targets, up, low

    def means_reg_find_inducing_inputs(self, data_points, target_values, num_inputs, max_iter=None):
        """
        Find inducing inputs for gp-regression using variational learning approach
        :param data_points: data points
        :param target_values: target values at data points
        :param num_inputs: number of inducing inputs to be found
        :param max_iter: maximum number of iterations
        :return:
        """
        if not(isinstance(data_points, np.ndarray) and
               isinstance(target_values, np.ndarray)):
            raise TypeError("The operands must be of type numpy array")

        dim = data_points.shape[0]
        param_len = self.covariance_obj.get_params().size

        def loc_fun(w):
            # print("Params:", w[:param_len])
            loss, grad = self._means_reg_inducing_points_oracle(data_points, target_values, w, inputs)
            return -loss, -grad

        bnds = self.covariance_obj.get_bounds()
        if max_iter is None:
            max_iter = np.inf
        means = KMeans(n_clusters=num_inputs)
        means.fit(data_points.T)
        inputs = means.cluster_centers_.T
        np.random.seed(15)
        w0 = self.covariance_obj.get_params()
        # f2, g2 = loc_fun(w0)
        # print("Gradient:", g2)
        # for i in range(w0.size):
        #     diff = np.zeros(w0.shape)
        #     diff[i] = 1e-8
        #     f1, g1 = loc_fun(w0 + diff)
        #     print("Difference:", (f1 - f2) * 1e8)
        # exit(0)
        res, w_list, time_list, fun_lst = minimize_wrapper(loc_fun, w0, method='L-BFGS-B',
                                                  mydisp=True, bounds=bnds, options={'gtol': 1e-8, 'ftol': 0,
                                                                                     'maxiter': max_iter})
        optimal_params = res.x
        print(res)
        print(inputs.shape)
        self.covariance_obj.set_params(optimal_params)

        cov_fun = self.covariance_obj.covariance_function
        sigma = optimal_params[-1]
        K_mm = cov_fun(inputs, inputs)
        K_mn = cov_fun(inputs, data_points)
        K_nm = K_mn.T
        Sigma = np.linalg.inv(K_mm + K_mn.dot(K_nm)/sigma**2)
        mu = sigma**(-2) * K_mm.dot(Sigma).dot(K_mn).dot(target_values)
        A = K_mm.dot(Sigma).dot(K_mm)
        return inputs, mu, A, w_list, time_list, fun_lst

    def _means_reg_inducing_points_oracle(self, points, targets, params, ind_points):
        """
        :param points: data points array
        :param targets: target values vector
        :param params: hyper-parameters vector
        :param ind_points: inducing points
        """
        sigma = params[-1]
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        K_mm = cov_fun(ind_points, ind_points)
        K_mm_inv = np.linalg.inv(K_mm) # use cholesky?
        K_nm = cov_fun(points, ind_points)
        K_mn = K_nm.T
        Q_nn = (K_nm.dot(K_mm_inv)).dot(K_mn)
        B = Q_nn + np.eye(Q_nn.shape[0]) * sigma**2
        B_l = np.linalg.cholesky(B)
        B_l_inv = np.linalg.inv(B_l)
        B_inv = B_l_inv.T.dot(B_l_inv)
        zero = np.array([[0]])
        K_nn_diag = cov_fun(zero, zero)
        F_v = - np.sum(np.log(np.diag(B_l))) - targets.T.dot(B_inv).dot(targets)/2 - \
              np.sum(K_nn_diag - np.diag(Q_nn)) / (2 * sigma**2)

        # Gradient
        gradient = []
        derivative_matrix_list = cov_obj.get_derivative_function_list(params)
        for func in derivative_matrix_list:
            dK_nm = func(points, ind_points)
            dK_mn = dK_nm.T
            dK_mm = func(ind_points, ind_points)
            dK_mm_inv = - K_mm_inv.dot(dK_mm.dot(K_mm_inv))
            dB_dtheta = (dK_nm.dot(K_mm_inv) + K_nm.dot(dK_mm_inv)).dot(K_mn) + K_nm.dot(K_mm_inv.dot(dK_mn))
            dK_nn = func(zero, zero)
            gradient.append(self._reg_get_lower_bound_partial_derivative(targets, dB_dtheta, dB_dtheta, B_inv, sigma,
                                                                         dK_nn))

        # sigma derivative
        dK_mm = cov_obj.get_noise_derivative(K_mm.shape[0])
        dK_mm_inv = - K_mm_inv.dot(dK_mm.dot(K_mm_inv))
        dQ_dtheta = K_nm.dot(dK_mm_inv).dot(K_mn)
        dB_dtheta = dQ_dtheta + 2 * sigma * np.eye(Q_nn.shape[0])
        dK_nn = 2 * sigma
        gradient.append(self._reg_get_lower_bound_partial_derivative(targets, dB_dtheta, dQ_dtheta, B_inv, sigma,
                                                                     dK_nn))
        gradient[-1] += np.sum(K_nn_diag - np.diag(Q_nn)) / sigma**3
        return F_v[0, 0], np.array(gradient)
