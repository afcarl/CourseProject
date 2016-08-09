import copy
import time

import numpy as np
import scipy as sp
import scipy.optimize as op
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from warnings import warn
from numpy.polynomial.hermite import hermgauss
from copy import deepcopy
from scipy.special import expit

from GP.covariance_functions import CovarianceFamily, sigmoid
from GP.gaussian_process import GP
from GP.optimization import check_gradient, minimize_wrapper, stochastic_gradient_descent, climin_adadelta_wrapper, \
    gradient_descent
from GP.gp_res import GPRes

class GPC(GP):
    """
    Gaussian Process Classifier
    """

    def __init__(self, cov_obj, mean_function=lambda x: 0, method='brute', hermgauss_deg=100):
        """
        :param cov_obj: object of the CovarianceFamily class
        :param mean_function: function, mean of the gaussian process
        :param method: method
            'brute' — Laplace approximation method
            'brute_alt' — A slightly different approach to maximizing the evidence in brute method
            'svi' — inducing input method from Scalable Variational Gaussian Process Classification article
            'vi' — experimental inducing input method, similar to `vi` method for regression
        :param hermgauss_deg: degree of Gussian-hermite quadrature, used for svi method only
        :return: GPR object
        """
        if not isinstance(cov_obj, CovarianceFamily):
            raise TypeError("The covariance object cov_obj is of the wrong type")
        if not hasattr(mean_function, '__call__'):
            raise TypeError("mean_function must be callable")

        self.covariance_fun = cov_obj.covariance_function
        self.covariance_obj = cov_obj
        self.mean_fun = mean_function
        self.method = method

        # A tuple: inducing inputs, and parameters of gaussian distribution at these points (mean and covariance)
        self.inducing_inputs = None
        # A tuple: weights and points for Gauss-Hermite quadrature
        self.gauss_hermite = None
        self.hermgauss_degree = hermgauss_deg
        if self.method == 'svi':
            self._svi_gauss_hermite_precompute()

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
        targets = np.sign(targets)
        targets = targets.reshape((targets.size, 1))
        return targets[:tr_points.shape[1], :], targets[tr_points.shape[1]:, :]

    @staticmethod
    def _brute_get_ml(f_opt, b, cov_inv):
        """
        :param f_opt: posterior mode
        :param b: I + W^(1/2) K W^(1/2), where W is minus hessian of the log likelihood at f_opt
        :param cov_inv: inverse covariance matrix at data points
        :return: the Laplace estimation of log marginal likelihood
        """
        l_b = np.linalg.cholesky(b)
        return -((f_opt.T.dot(cov_inv)).dot(f_opt) + 2 * np.sum(np.log(np.diag(l_b))))/2

    @staticmethod
    def _brute_get_b(points_cov, hess_opt):
        """
        :param points_cov: covariance matrix
        :param hess_opt: -\nabla\nabla log p(y|f)|_{f_opt}
        :return: I + W^(1/2) K W^(1/2), where W is minus hessian of the log likelihood at f_opt
        """
        w = sp.linalg.sqrtm(hess_opt)
        b = (w.dot(points_cov)).dot(w)
        return np.eye(b.shape[0]) + b

    @staticmethod
    def _brute_get_laplace_approximation(labels, cov_inv, cov_l, max_iter=1000):
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
    def _brute_get_ml_partial_derivative(f_opt, k_inv, ancillary_mat, dk_dtheta_mat):
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

    def _brute_get_ml_grad(self, points, cov_inv, f_opt, anc_mat, params):
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
        return np.array([self._brute_get_ml_partial_derivative(f_opt, cov_inv, anc_mat,
                                                               func(points, points))
                         for func in derivative_matrix_list] +
                        [self._brute_get_ml_partial_derivative(f_opt, cov_inv, anc_mat,  noise_derivative)])

    def _brute_oracle(self, points, f_opt, hess_opt, params):
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

        b_mat = self._brute_get_b(points_cov, hess_opt)
        marginal_likelihood = self._brute_get_ml(f_opt, b_mat, points_cov_inv)

        anc_mat = np.linalg.inv(np.linalg.inv(hess_opt) + points_cov)
        gradient = self._brute_get_ml_grad(points, points_cov_inv, f_opt, anc_mat, params)

        return marginal_likelihood, gradient

    def _brute_fit(self, points, labels, max_iter=10, alternate=False):
        """
        optimizes the self.covariance_obj hyper-parameters
        :param points: data points
        :param labels: class labels at data points
        :param max_iter: maximim number of iterations
        :return: a list of hyper-parameters values at different iterations and a list of times iteration-wise
        """
        if alternate:
            return self._brute_alternative_find_hyper_parameters(points, labels, max_iter)
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_fun = cov_obj.covariance_function
        bnds = self.covariance_obj.get_bounds()
        w0 = self.covariance_obj.get_params()
        w_list = []
        time_list = []

        def func(w):
            loss, grad = self._brute_oracle(points, f_opt, hess_opt, w)
            return -loss, -grad

        start = time.time()
        for i in range(max_iter):
            points_cov = cov_fun(points, points)
            points_l = np.linalg.cholesky(points_cov)
            points_l_inv = np.linalg.inv(points_l)
            points_cov_inv = points_l_inv.T.dot(points_l_inv)

            f_opt, hess_opt = self._brute_get_laplace_approximation(labels, points_cov_inv, points_l, max_iter=20)

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

    def _brute_predict(self, test_points, training_points, training_labels):
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
        f_opt, hess_opt = self._brute_get_laplace_approximation(training_labels, points_cov_inv, points_l)
        k_test_x = cov_fun(test_points, training_points)

        f_test = np.dot(np.dot(k_test_x, points_cov_inv), f_opt)
        loc_y_test = np.sign(f_test.reshape((f_test.size, 1)))

        return loc_y_test

    def predict(self, *args, **kwargs):
        if self.method == 'brute' or self.method == 'brute_alt':
            return self._brute_predict(*args, **kwargs)
        elif self.method == 'svi' or self.method == 'vi':
            return self._inducing_points_predict(*args, **kwargs)
        else:
            raise ValueError("Unknown method")

    def fit(self, *args, **kwargs):
        if self.method == 'brute':
            return self._brute_fit(*args, **kwargs)
        elif self.method == 'brute_alt':
            return self._brute_alternative_fit(*args, **kwargs)
        elif self.method == 'svi':
            return self._svi_fit(*args, **kwargs)
        elif self.method == 'vi':
            options = kwargs ['optimizer_options']
            if 'bound' in options.keys():
                if options['bound'] == 'JJ':
                    del options['bound']
                    return self._vi_jj_fit(*args, **kwargs)
                elif options['bound'] == 'Taylor':
                    del options['bound']
                    return self._vi_taylor_fit(*args, **kwargs)
                else:
                    del options['bound']
                    warn('Unknown bound for vi method')
                    return self._vi_jj_fit(*args, **kwargs)
            else:
                return self._vi_jj_fit(*args, **kwargs)

        else:
            print(self.method)
            raise ValueError("Method " + self.method + " is invalid")

    @staticmethod
    def _brute_get_implicit_ml_partial_derivative(f_opt, k_inv, ancillary_mat, dk_dtheta_mat, labels):
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

    def _brute_get_ml_full_grad(self, points, labels, cov_inv, f_opt, anc_mat, params):
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

        return np.array([self._brute_get_implicit_ml_partial_derivative(f_opt, cov_inv, anc_mat,
                                                                        func(points, points), labels) +
                         self._brute_get_ml_partial_derivative(f_opt, cov_inv, anc_mat,
                                                               func(points, points))
                         for func in derivative_matrix_list] +
                        [self._brute_get_implicit_ml_partial_derivative(f_opt, cov_inv, anc_mat,  noise_derivative,
                                                                        labels) +
                         self._brute_get_ml_partial_derivative(f_opt, cov_inv, anc_mat,  noise_derivative)])

    @staticmethod
    def _brute_get_full_ml(f_opt, b, cov_inv, labels):
        """
        :param f_opt: posterior mode
        :param b: I + W^(1/2) K W^(1/2), where W is minus hessian of the log likelihood at f_opt
        :param cov_inv: inverse covariance matrix at data points
        :return: the Laplace estimation of log marginal likelihood
        """
        l_b = np.linalg.cholesky(b)
        return -(((f_opt.T.dot(cov_inv)).dot(f_opt) + 2 * np.sum(np.log(np.diag(l_b))))/2 +
                 np.sum(np.log(1 + np.exp(-labels * f_opt))))

    def _brute_alternative_oracle(self, points, labels, f_opt, hess_opt, params):
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

        b_mat = self._brute_get_b(points_cov, hess_opt)
        marginal_likelihood = self._brute_get_full_ml(f_opt, b_mat, points_cov_inv, labels.reshape(labels.size, ))

        anc_mat = np.linalg.inv(np.linalg.inv(hess_opt) + points_cov)
        gradient = self._brute_get_ml_full_grad(points, labels.reshape(labels.size, ), points_cov_inv,
                                                f_opt, anc_mat, params)

        return marginal_likelihood, gradient

    def _brute_alternative_fit(self, points, labels, max_iter=10):
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
            loss, _ = self._brute_alternative_oracle(points, labels, f_opt, hess_opt, w)
            return -loss

        def grad(w):
            _, gradient = self._brute_alternative_oracle(points, labels, f_opt, hess_opt, w)
            return -gradient

        start = time.time()

        for i in range(max_iter):
            points_cov = cov_fun(points, points)
            points_l = np.linalg.cholesky(points_cov)
            points_l_inv = np.linalg.inv(points_l)
            points_cov_inv = points_l_inv.T.dot(points_l_inv)
            # det_k = 2 * np.sum(np.log(np.diag(points_l)))

            f_opt, hess_opt = self._brute_get_laplace_approximation(labels, points_cov_inv, points_l, max_iter=np.inf)
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

    def _svi_get_parameter_vector(self, theta, eta_1, eta_2):
        """
        Transform given parameters to a vector, according to the used parametrization
        :param theta: vector or list
        :param eta_1: vector
        :param eta_2: matrix
        :return: a vector
        """
        theta = np.array(theta).reshape(-1)[:, None]
        eta_1 = eta_1.reshape(-1)[:, None]
        eta_2 = self._svi_lower_triang_mat_to_vec(eta_2)[:, None]
        return np.vstack((theta, eta_1, eta_2))[:, 0]

    def _svi_get_parameters(self, vec):
        """
        Retrieve the parameters from the parameter vector vec of the same form, as the one, _svi_get_parameter_vector
        produces.
        :param vec: a vector
        :return: a tuple of parameters
        """
        theta_len = len(self.covariance_obj.get_params())
        vec = vec[:, None]
        theta = vec[:theta_len, :]
        mat_size = np.int(np.sqrt(2 * (vec.size - theta_len) + 9 / 4) - 3 / 2)
        eta_1 = vec[theta_len:theta_len+mat_size, :].reshape((mat_size, 1))
        eta_2 = self._svi_lower_triang_vec_to_mat(vec[theta_len+mat_size:, :])
        return theta, eta_1, eta_2

    def _svi_get_bounds(self, m):
        bnds = list(self.covariance_obj.get_bounds())
        bnds += [(None, None)] * m
        sigma_bnds = self._svi_lower_triang_mat_to_vec(np.eye(m)).tolist()
        for elem in sigma_bnds:
            if elem == 0:
                bnds.append((None, None))
            else:
                bnds.append((1e-2, None))
        return tuple(bnds)

    def _svi_fit(self, data_points, target_values, num_inputs=0, inputs=None, optimizer_options={}):
        """
        A method for optimizing hyper-parameters (for fixed inducing points), based on stochastic variational inference
        :param data_points: training set objects
        :param target_values: training set answers
        :param inputs: inducing inputs
        :param num_inputs: number of inducing points to generate. If inducing points are provided, this parameter is
        ignored
        :param optimizer_options: options for the optimization method
        :return:
        """

        # if no inducing inputs are provided, we use K-Means cluster centers as inducing inputs
        if inputs is None:
            means = KMeans(n_clusters=num_inputs)
            means.fit(data_points.T)
            inputs = means.cluster_centers_.T

        # Initializing required variables
        y = target_values
        m = inputs.shape[1]
        n = y.size


        # Initializing variational (normal) distribution parameters
        mu = np.zeros((m, 1))
        sigma_L = np.eye(m)  # Cholesky factor of sigma

        theta = self.covariance_obj.get_params()
        param_vec = self._svi_get_parameter_vector(theta, mu, sigma_L)

        bnds = self._svi_get_bounds(m)

        def fun(w, i=None):
            full = False
            if i is None:
                full = True
                i = range(target_values.size)
            fun, grad = self._svi_elbo_batch_approx_oracle(data_points, target_values, inputs, parameter_vec=w,
                                       indices=i)
            if full:
                return -fun, -grad[:, 0]
            else:
                return -grad[:, 0]

        def adadelta_fun(w, train_points, train_targets):
            _, grad = self._svi_elbo_batch_approx_oracle(train_points, train_targets, inputs, parameter_vec=w,
                                       indices=range(train_targets.size), N=n)
            return -grad[:, 0]

        mydisp = False
        mode = 'full'

        opts = copy.deepcopy(optimizer_options)
        if not optimizer_options is None:
            if 'mydisp' in opts.keys():
                mydisp = opts['mydisp']
                del opts['mydisp']
            if 'mode' in opts.keys():
                mode = opts['mode']
                del opts['mode']

        if mode == 'full':
            res, w_list, time_list = minimize_wrapper(fun, param_vec, method='L-BFGS-B', mydisp=mydisp,
                                                      bounds=bnds, jac=True, options=opts)
            res = res['x']
        elif mode == 'batch':
            res, w_list, time_list = stochastic_gradient_descent(oracle=fun, n=n, point=param_vec, bounds=bnds,
                                                                 options=opts)
        elif mode == 'adadelta':
            res, w_list, time_list = climin_adadelta_wrapper(oracle=adadelta_fun, w0=param_vec, train_points=data_points,
                                                             train_targets=target_values, options=opts)

        theta, mu, sigma_L = self._svi_get_parameters(res)
        sigma = sigma_L.dot(sigma_L.T)

        self.inducing_inputs = (inputs, mu, sigma)
        return GPRes(param_lst=w_list, time_lst=time_list)

    def _svi_get_prediction_quality(self, params, test_points, test_targets):
        """
        Returns prediction quality on the test set for the given kernel (and inducing points) parameters for the means
        method
        :param params: parameters
        :param test_points: test set points
        :param test_targets: test set target values
        :return: prediction MSE
        """
        new_gp = deepcopy(self)
        theta, mu, Sigma_L = new_gp._svi_get_parameters(params)
        Sigma = Sigma_L.dot(Sigma_L.T)
        # theta = params[:len(new_gp.covariance_obj.get_params())]
        new_gp.covariance_obj.set_params(theta)
        new_gp.inducing_inputs = (new_gp.inducing_inputs[0], mu, Sigma)
        predicted_y_test = new_gp.predict(test_points)
        return 1 - np.sum(test_targets != predicted_y_test) / test_targets.size
        # return f1_score(test_targets, predicted_y_test)

    def _svi_elbo_batch_approx_oracle(self, data_points, target_values, inducing_inputs, parameter_vec,
                                       indices, N=None):
        """
        The approximation of Evidence Lower Bound (L3 from the article 'Scalable Variational Gaussian Process
        Classification') and it's derivative wrt kernel hyper-parameters and variational parameters.
        The approximation is found using a mini-batch.
        :param data_points: the array of data points
        :param target_values: the target values at these points
        :param inducing_inputs: an array of inducing inputs
        :param mu: the current mean of the process at the inducing points
        :param sigma: the current covariance of the process at the inducing points
        :param theta: a vector of hyper-parameters and variational parameters, the point of evaluation
        :param indices: a list of indices of the data points in the mini-batch
        :return: ELBO and it's gradient approximation in a tuple
        """
        if N is None:
            N = target_values.size
        m = inducing_inputs.shape[1]
        theta, mu, sigma_L = self._svi_get_parameters(parameter_vec)
        sigma = sigma_L.dot(sigma_L.T)
        old_params = self.covariance_obj.get_params()
        self.covariance_obj.set_params(theta)

        l = len(indices)
        i = indices
        y_i = target_values[i]
        x_i = data_points[:, i]

        # Covariance function and it's parameters
        cov_fun = self.covariance_fun
        params = self.covariance_obj.get_params()
        sigma_n = parameter_vec[len(self.covariance_obj.get_params())-1]

        # Covariance matrices

        K_mm = cov_fun(inducing_inputs, inducing_inputs)
        try:
            L = np.linalg.cholesky(K_mm)
        except:
            print(params)
            exit(0)

        L_inv = np.linalg.inv(L)
        K_mm_inv = L_inv.T.dot(L_inv)
        k_i = cov_fun(inducing_inputs, x_i)
        # Lambda_i = K_mm_inv.dot(k_i.dot(k_i.T.dot(K_mm_inv))) / sigma_n**2
        K_ii = cov_fun(x_i[:, :1], x_i[:, :1])
        # tilde_K_ii = l * K_ii - np.einsum('ij,ji->', k_i.T, K_mm_inv.dot(k_i))

        # Derivatives
        derivative_matrix_list = self.covariance_obj.get_derivative_function_list(params)
        d_K_mm__d_theta_lst = [fun(inducing_inputs, inducing_inputs) for fun in derivative_matrix_list]
        d_k_i__d_theta_lst = [fun(inducing_inputs, data_points[:, i]) for fun in derivative_matrix_list]
        d_K_mm__d_sigma_n = self.covariance_obj.get_noise_derivative(points_num=m)
        d_k_i__d_sigma_n = np.zeros((m, l))
        d_K_mm__d_theta_lst.append(d_K_mm__d_sigma_n)
        d_k_i__d_theta_lst.append(d_k_i__d_sigma_n)


        #NEW STUFF
        #f_i marginal distribution parameters

        m_i = k_i.T.dot(K_mm_inv.dot(mu))
        S_i = np.sqrt((K_ii + np.einsum('ij,ji->i', k_i.T, K_mm_inv.dot((sigma - K_mm).dot(K_mm_inv.dot(k_i))))).T)

        # Variational Lower Bound, estimated by the mini-batch
        loss = self._svi_compute_log_likelihood_expectation(m_i, S_i, y_i)
        loss += - np.sum(np.log(np.diag(L))) * l / N
        loss += np.sum(np.log(np.diag(sigma_L))) * l / N
        loss += - np.trace(sigma.dot(K_mm_inv)) * l / (2*N)
        loss += - mu.T.dot(K_mm_inv.dot(mu)) * l / (2*N)


        # Gradient
        grad = np.zeros((len(theta,)))
        mu_expectations = self._svi_compute_mu_grad_expectation(m_i, S_i, y_i)
        dL_dmu = (np.sum(mu_expectations * K_mm_inv.dot(k_i), axis=1)[:, None]
                  - K_mm_inv.dot(mu) * l / N)

        sigma_expectations = self._svi_compute_sigma_l_grad_expectation(m_i, S_i, y_i)
        dL_dsigma_L = K_mm_inv.dot((k_i * sigma_expectations).dot(k_i.T.dot(K_mm_inv.dot(sigma_L)))) + \
                      np.eye(m) * l / (N * np.diag(sigma_L)) - (K_mm_inv.dot(sigma_L)) * l / N
        dL_dsigma_L = self._svi_lower_triang_mat_to_vec(dL_dsigma_L)

        for param in range(len(theta)):
            if param != len(theta) - 1:
                cov_derivative = derivative_matrix_list[param]
            else:
                cov_derivative = lambda x, y: self.covariance_obj.get_noise_derivative(points_num=1)

            d_K_mm__d_theta = d_K_mm__d_theta_lst[param]
            d_k_i__d_theta = d_k_i__d_theta_lst[param]
            grad[param] += np.einsum('ij,ji->', (d_k_i__d_theta * mu_expectations).T, K_mm_inv.dot(mu))
            grad[param] += - np.einsum('ij,ji->', (k_i * mu_expectations).T, K_mm_inv.dot(d_K_mm__d_theta.
                                                                                        dot(K_mm_inv.dot(mu))))
            grad[param] += cov_derivative(x_i[:, :1], x_i[:, :1]) * np.sum(sigma_expectations) / 2

            grad[param] += np.einsum('ij,ji->', d_k_i__d_theta.T, K_mm_inv.dot((sigma_L.dot(sigma_L.T) - K_mm).dot(
                K_mm_inv.dot(k_i * sigma_expectations))))
            grad[param] += - np.einsum('ij,ji->', k_i.T, K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv.dot(
                (sigma_L.dot(sigma_L.T) - K_mm).dot(K_mm_inv.dot(k_i * sigma_expectations))))))
            grad[param] += - 1/2 * np.einsum('ij,ji->', k_i.T, K_mm_inv.dot(d_K_mm__d_theta.dot(
                K_mm_inv.dot(k_i * sigma_expectations))))
            grad[param] += - np.trace(K_mm_inv.dot(d_K_mm__d_theta)) * l / (2*N)
            grad[param] += np.trace(sigma.dot(K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv)))) * l / (2*N)
            grad[param] += mu.T.dot(K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv.dot(mu)))) * l / (2*N)

        grad = grad[:, None]
        grad = np.vstack((grad, dL_dmu.reshape(-1)[:, None]))
        grad = np.vstack((grad, dL_dsigma_L.reshape(-1)[:, None]))
        return loss, grad

    def _svi_gauss_hermite_precompute(self):
        """
        Precompute weights and points for Gauss-Hermite quadrature
        :return: None
        """
        points, weights = hermgauss(self.hermgauss_degree)
        self.gauss_hermite = points, weights

    def _svi_compute_expectation(self, variable, mu, sigma):
        """
        Computes the approximate expectation of a one-dimensional random variable with respect to a normal distribution
        with given parameters, using gauss-hermite quadrature.
        :param variable: the random variable under expectation
        :param mu: mean of the distribution
        :param sigma: std of the distribution
        :return: expectation
        """
        points, weights = self.gauss_hermite
        expectation = 0
        sqrt_two = np.sqrt(2)
        for weight, point in zip(weights, points):
            expectation += weight * variable(sqrt_two * point * sigma + mu)
        return expectation / np.sqrt(np.pi)

    def _svi_compute_log_likelihood_expectation(self, means, stds, targets):
        points, weights = self.gauss_hermite
        points = points[None, :]
        weights = weights[None, :]
        mat = - (np.sqrt(2) * points * stds + means) * targets
        # mat = np.exp(mat)
        # mat += 1
        # mat = -np.log(mat)
        # print(np.min(mat))
        # print(np.max(mat))
        mat = np.log(expit(np.abs(mat))) - mat * (np.sign(mat) == 1)
        # mat = np.log(expit(mat)) - mat
        mat *= weights
        return np.sum(mat) / np.sqrt(np.pi)

    def _svi_compute_mu_grad_expectation(self, means, stds, targets):
        points, weights = self.gauss_hermite
        points = points[None, :]
        weights = weights[None, :]
        # mat = targets / (1 + np.exp(targets * (np.sqrt(2) * points * stds + means)))
        mat = targets * (np.sqrt(2) * points * stds + means)
        mat = expit(-mat) * targets
        mat *= weights
        return (np.sum(mat, axis=1) / np.sqrt(np.pi))[None, :]

    def _svi_compute_sigma_l_grad_expectation(self, means, stds, targets):
        points, weights = self.gauss_hermite
        points = points[None, :]
        weights = weights[None, :]
        # anc_mat = np.exp(targets * (np.sqrt(2) * points * stds + means))
        # mat = -targets**2 * anc_mat / (1 + anc_mat)**2
        anc_mat = targets * (np.sqrt(2) * points * stds + means)
        mat = -targets**2 * expit(anc_mat) * expit(-anc_mat)
        mat *= weights
        return (np.sum(mat, axis=1) / np.sqrt(np.pi))[None, :]

    def _inducing_points_predict(self, test_points):
        """
        Predict new values given inducing points
        :param ind_points: inducing points
        :param expectation: expectation at inducing points
        :param covariance: covariance at inducing points
        :param test_points: test points
        :return: predicted values at inducing points
        """
        ind_points, expectation, covariance = self.inducing_inputs
        cov_fun = self.covariance_obj.covariance_function
        K_xm = cov_fun(test_points, ind_points)
        K_mx = K_xm.T
        K_mm = cov_fun(ind_points, ind_points)
        K_xx = cov_fun(test_points, test_points)
        K_mm_inv = np.linalg.inv(K_mm)

        new_mean = K_xm.dot(K_mm_inv).dot(expectation)
        new_cov = K_xx - K_xm.dot(K_mm_inv).dot(K_mx) + K_xm.dot(K_mm_inv).dot(covariance).dot(K_mm_inv).dot(K_mx)

        test_targets, up, low = self.sample_for_matrices(new_mean, new_cov)
        return np.sign(test_targets)

    def _vi_jj_recompute_xi(self, K_mm, K_mm_inv, K_nm, K_ii, mu, Sigma):
        K_mn = K_nm.T
        means = K_nm.dot(K_mm_inv.dot(mu))
        vars = K_ii + np.einsum('ij,ji->i', K_nm, K_mm_inv.dot((Sigma - K_mm).dot(K_mm_inv.dot(K_mn))))[:, None]
        return np.sqrt(means**2 + vars)

    def _vi_jj_lambda(self, xi):
        return np.tanh(xi / 2) / (4 * xi)

    def _vi_jj_recompute_var_parameters(self, K_mm_inv, K_nm, xi, y):
        K_mn = K_nm.T
        Lambda_xi = self._vi_jj_lambda(xi)
        Sigma = np.linalg.inv(2 * K_mm_inv.dot(K_mn.dot((Lambda_xi * K_nm).dot(K_mm_inv))) + K_mm_inv)
        mu = Sigma.dot(K_mm_inv.dot(K_mn.dot(y))) / 2
        return mu, Sigma

    def _vi_jj_fit(self, data_points, target_values, num_inputs=0, inputs=None, max_out_iter=20, optimizer_options={}):
        """
        An experimental method for optimizing hyper-parameters (for fixed inducing points), based on variational
        inference and Jaakkola-Jordan lower bound for logistic function. See review.
        :param data_points: training set objects
        :param target_values: training set answers
        :param inputs: inducing inputs
        :param num_inputs: number of inducing points to generate. If inducing points are provided, this parameter is
        ignored
        :param optimizer_options: options for the optimization method
        :return:
        """
        # if no inducing inputs are provided, we use K-Means cluster centers as inducing inputs
        if inputs is None:
            means = KMeans(n_clusters=num_inputs)
            means.fit(data_points.T)
            inputs = means.cluster_centers_.T

        # Initializing required variables
        y = target_values
        m = inputs.shape[1]

        # Initializing variational (normal) distribution parameters
        mu = np.zeros((m, 1), dtype=float)
        Sigma = np.eye(m)

        def oracle(x):
            fun, grad = self._vi_jj_elbo(data_points, target_values, x, inputs, xi)
            return -fun, -grad

        bnds = self.covariance_obj.get_bounds()
        params = self.covariance_obj.get_params()
        w_list, time_list = [(params, mu, Sigma)], [0]
        start = time.time()
        num_updates = 3
        if 'num_updates' in optimizer_options.keys():
            num_updates = optimizer_options['num_updates']
            del optimizer_options['num_updates']

        mydisp = False
        options = copy.deepcopy(optimizer_options)
        if not optimizer_options is None:
            if 'mydisp' in optimizer_options.keys():
                mydisp = optimizer_options['mydisp']
                del options['mydisp']

        for iteration in range(max_out_iter):
            xi, mu, Sigma = self._vi_jj_update_xi(params, data_points, target_values, inputs, mu, Sigma,
                                                  n_iter=num_updates)

            it_res, it_w_list, it_time_list = minimize_wrapper(oracle, params, method='L-BFGS-B', mydisp=mydisp,
                                                               bounds=bnds, options=options)

            params = it_res['x']

            w_list.append((params, np.copy(mu), np.copy(Sigma)))
            time_list.append(time.time() - start)
            if mydisp:
                print('\tHyper-parameters at outter iteration', iteration, ':', params)
        self.inducing_inputs = inputs, mu, Sigma
        self.covariance_obj.set_params(params)
        return GPRes(param_lst=w_list, time_lst=time_list)

    def _vi_jj_update_xi(self, params, data_points, target_values, inputs, mu, Sigma, n_iter=5):
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        K_nm = cov_fun(data_points, inputs)
        K_mm = cov_fun(inputs, inputs)
        K_mm_inv, K_log_det = self._get_inv_logdet_cholesky(K_mm)
        K_ii = cov_fun(data_points[:, :1], data_points[:, :1])
        for i in range(n_iter):
            xi = self._vi_jj_recompute_xi(K_mm, K_mm_inv, K_nm, K_ii, mu, Sigma)
            mu, Sigma = self._vi_jj_recompute_var_parameters(K_mm_inv, K_nm, xi, target_values)
        return xi, mu, Sigma

    def _vi_jj_elbo(self, points, targets, params, ind_points, xi):
        """
        The evidence lower bound, used in the vi method.
        :param points: data points
        :param targets: target values
        :param params: hernel hyper-parameters
        :param ind_points: inducing input positions
        :param xi: variational parameters xi
        :return: the value and the gradient of the lower bound
        """
        y = targets
        n = points.shape[1]
        m = ind_points.shape[1]
        sigma = params[-1]
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        lambda_xi = self._vi_jj_lambda(xi)
        K_mm = cov_fun(ind_points, ind_points)
        K_mm_inv, K_mm_log_det = self._get_inv_logdet_cholesky(K_mm)
        K_nm = cov_fun(points, ind_points)
        K_mn = K_nm.T
        K_mnLambdaK_nm = K_mn.dot(lambda_xi*K_nm)
        K_ii = cov_fun(points[:, :1], points[:, :1])

        B = 2 * K_mnLambdaK_nm + K_mm

        B_inv, B_log_det = self._get_inv_logdet_cholesky(B)



        fun = ((y.T.dot(K_nm.dot(B_inv.dot(K_mn.dot(y))))/8)[0, 0] + K_mm_log_det/2 - B_log_det/2
               - np.sum(K_ii * lambda_xi) + np.einsum('ij,ji->', K_mm_inv, K_mnLambdaK_nm))

        gradient = []
        derivative_matrix_list = cov_obj.get_derivative_function_list(params)
        # for func in derivative_matrix_list:
        for param in range(len(params)):
            if param != len(params) - 1:
                func = derivative_matrix_list[param]
            else:
                func = lambda x, y: cov_obj.get_noise_derivative(points_num=1)
            if param != len(params) - 1:
                dK_mm = func(ind_points, ind_points)
                dK_nm = func(points, ind_points)
                dK_mn = dK_nm.T
                dB = 4 * dK_mn.dot(lambda_xi*K_nm) + dK_mm
            else:
                dK_mm = np.eye(m) * func(ind_points, ind_points)
                dK_mn = np.zeros_like(K_mn)
                dK_nm = dK_mn.T
                dB = dK_mm
            dK_nn = func(np.array([[0]]), np.array([[0]]))
            derivative = np.array([[0]], dtype=float)
            derivative += y.T.dot(dK_nm.dot(B_inv.dot(K_mn.dot(y))))/4
            derivative -= y.T.dot(K_nm.dot(B_inv.dot(dB.dot(B_inv.dot(K_mn.dot(y))))))/8
            derivative += np.trace(K_mm_inv.dot(dK_mm))/2
            derivative -= np.trace(B_inv.dot(dB))/2
            derivative -= np.sum(lambda_xi * dK_nn)
            derivative += np.trace(2 * K_mm_inv.dot(K_mn.dot(lambda_xi*dK_nm)) -
                                   K_mm_inv.dot(dK_mm.dot(K_mm_inv.dot(K_mnLambdaK_nm))))
            gradient.append(derivative[0, 0])
        return fun, np.array(gradient)

    def _vi_jj_get_prediction_quality(self, params, test_points, test_targets):
        """
        Returns prediction quality on the test set for the given kernel (and inducing points) parameters for the means
        method
        :param params: parameters
        :param test_points: test set points
        :param test_targets: test set target values
        :return: prediction MSE
        """
        new_gp = deepcopy(self)
        theta, mu, Sigma = params
        new_gp.covariance_obj.set_params(theta)
        new_gp.inducing_inputs = (new_gp.inducing_inputs[0], mu, Sigma)
        predicted_y_test = new_gp.predict(test_points)
        # return f1_score(test_targets, predicted_y_test)
        return 1 - np.sum(test_targets != predicted_y_test) / test_targets.size

    def get_prediction_quality(self, *args, **kwargs):

        if self.method == 'vi':
            # raise ValueError('Not implemented yet')
            return self._vi_jj_get_prediction_quality(*args, **kwargs)
        elif self.method == 'svi':
            return self._svi_get_prediction_quality(*args, **kwargs)
        else:
            raise ValueError('Wrong method')

    #### VI Taylor
    def _vi_taylor_phi(self, xi, y):
        return y * expit(-y * xi)
        # return y / (1 + np.exp(y * xi))

    def _vi_taylor_psi(self, xi, y):
        return expit(y * xi) * expit(-y * xi)
        # return (1 / (1 + np.exp(y * xi))) / (1 + np.exp(-y * xi))

    def _vi_taylor_v(self, xi, y):
        return self._vi_taylor_phi(xi, y) + 2 * self._vi_taylor_psi(xi, y) * xi

    def _vi_taylor_update_xi(self, params, data_points, target_values, inputs, mu, Sigma, n_iter=5):

        y = target_values
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        K_nm = cov_fun(data_points, inputs)
        K_mn = K_nm.T
        K_mm = cov_fun(inputs, inputs)
        K_mm_inv, K_log_det = self._get_inv_logdet_cholesky(K_mm)
        xi = 0

        for i in range(n_iter):
            xi = K_nm.dot(K_mm_inv.dot(mu))
            K_mnPsiK_nm = K_mn.dot(self._vi_taylor_psi(xi, y) * K_nm)
            K_mnPsiK_nm_inv, _ = self._get_inv_logdet_cholesky(K_mnPsiK_nm)
            B, _ = self._get_inv_logdet_cholesky(np.linalg.inv(K_mnPsiK_nm_inv) + 2 * K_mm_inv)
            Sigma = K_mm - 2 * B
            mu = Sigma.dot(K_mm_inv.dot(K_mn.dot(self._vi_taylor_v(xi, y))))

        return xi, mu, Sigma

    def _vi_taylor_fit(self, data_points, target_values, num_inputs=0, inputs=None, max_out_iter=20,
                       optimizer_options={}):
        """
        An experimental method for optimizing hyper-parameters (for fixed inducing points), based on variational
        inference and Second order Taylor approximation to the logistic function. See the review.
        :param data_points: training set objects
        :param target_values: training set answers
        :param inputs: inducing inputs
        :param num_inputs: number of inducing points to generate. If inducing points are provided, this parameter is
        ignored
        :param optimizer_options: options for the optimization method
        :return:
        """
        # if no inducing inputs are provided, we use K-Means cluster centers as inducing inputs
        if inputs is None:
            means = KMeans(n_clusters=num_inputs)
            means.fit(data_points.T)
            inputs = means.cluster_centers_.T

        # Initializing required variables
        m = inputs.shape[1]

        # Initializing variational (normal) distribution parameters
        mu = np.zeros((m, 1), dtype=float)
        Sigma = np.eye(m)

        def oracle(x):
            fun, grad = self._vi_taylor_elbo(data_points, target_values, x, inputs, xi)
            return -fun, -grad

        bnds = self.covariance_obj.get_bounds()
        params = self.covariance_obj.get_params()
        w_list, time_list = [(params, mu, Sigma)], [0]
        start = time.time()

        num_updates = 3
        if 'num_updates' in optimizer_options.keys():
            num_updates = optimizer_options['num_updates']
            del optimizer_options['num_updates']

        mydisp = False
        options = copy.deepcopy(optimizer_options)
        if not optimizer_options is None:
            if 'mydisp' in optimizer_options.keys():
                mydisp = optimizer_options['mydisp']
                del options['mydisp']

            # params = np.load('params.npy')
            # mu = np.load('mu.npy')
            # Sigma = np.load('sigma.npy')
            # xi = np.load('xi.npy')
            # print(params)
            # print(check_gradient(oracle, params, print_diff=True))
            # exit(0)
        for iteration in range(max_out_iter):
            xi, mu, Sigma = self._vi_taylor_update_xi(params, data_points, target_values, inputs, mu, Sigma,
                                                      n_iter=num_updates)

            it_res, it_w_list, it_time_list = minimize_wrapper(oracle, params, method='L-BFGS-B', mydisp=mydisp, bounds=bnds,
                                                               options=options)

            params = it_res['x']
            # print(check_gradient(oracle, params, print_diff=True))


            w_list.append((params, np.copy(mu), np.copy(Sigma)))
            time_list.append(time.time() - start)
            if mydisp:
                print('\tHyper-parameters at outter iteration', iteration, ':', params)
        self.inducing_inputs = inputs, mu, Sigma
        self.covariance_obj.set_params(params)
        return GPRes(param_lst=w_list, time_lst=time_list)

    def _vi_taylor_elbo(self, points, targets, params, ind_points, xi):
        """
        The evidence lower bound, used in the vi method.
        :param points: data points
        :param targets: target values
        :param params: hernel hyper-parameters
        :param ind_points: inducing input positions
        :param xi: variational parameters xi
        :return: the value and the gradient of the lower bound
        """
        y = targets
        m = ind_points.shape[1]
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        Psi_xi = self._vi_taylor_psi(xi, y)

        K_mm = cov_fun(ind_points, ind_points)
        K_mm_inv, K_mm_log_det = self._get_inv_logdet_cholesky(K_mm)
        K_nm = cov_fun(points, ind_points)
        K_mn = K_nm.T
        K_ii = cov_fun(points[:, :1], points[:, :1])

        K_mnPsiK_nm = K_mn.dot(Psi_xi*K_nm)
        K_mnPsiK_nm_inv, K_mnPsiK_nm_logdet = self._get_inv_logdet_cholesky(K_mnPsiK_nm)

        B_inv = K_mnPsiK_nm_inv + 2 * K_mm_inv
        B, B_log_det = self._get_inv_logdet_cholesky(B_inv)
        B_log_det *= -1

        v_xi = self._vi_taylor_v(xi, y)

        fun = 0
        fun += v_xi.T.dot(K_nm.dot(K_mm_inv.dot(K_mn.dot(v_xi)))) / 2
        fun += - v_xi.T.dot(K_nm.dot(K_mm_inv.dot(B.dot(K_mm_inv.dot(K_mn.dot(v_xi))))))
        fun += B_log_det / 2
        fun += - K_mnPsiK_nm_logdet / 2
        fun += - np.sum(Psi_xi) * K_ii
        fun += np.trace(K_mm_inv.dot(K_mnPsiK_nm))

        gradient = []
        derivative_matrix_list = cov_obj.get_derivative_function_list(params)
        for param in range(len(params)):
            if param != len(params) - 1:
                func = derivative_matrix_list[param]
            else:
                func = lambda x, y: cov_obj.get_noise_derivative(points_num=1)
            if param != len(params) - 1:
                dK_mm = func(ind_points, ind_points)
                dK_nm = func(points, ind_points)
                dK_mn = dK_nm.T
                d_K_mnPsiK_nm = 2 * dK_mn.dot(Psi_xi * K_nm)
                dB = (B.dot(K_mnPsiK_nm_inv.dot(d_K_mnPsiK_nm.dot(K_mnPsiK_nm_inv.dot(B)))) +
                          2 * B.dot(K_mm_inv.dot(dK_mm.dot(K_mm_inv.dot(B)))))
            else:
                dK_mm = np.eye(m) * func(ind_points, ind_points)
                dK_mn = np.zeros_like(K_mn)
                dK_nm = dK_mn.T
                d_K_mnPsiK_nm = np.zeros_like(K_mm)
                dB = 2 * B.dot(K_mm_inv.dot(dK_mm.dot(K_mm_inv.dot(B))))
            dK_nn = func(np.array([[0]]), np.array([[0]]))

            derivative = np.array([[0]], dtype=float)

            if param != len(params) - 1:
                derivative += v_xi.T.dot(dK_nm.dot(K_mm_inv.dot(K_mn.dot(v_xi))))

            derivative += - v_xi.T.dot(K_nm.dot(K_mm_inv.dot(dK_mm.dot(K_mm_inv.dot(K_mn.dot(v_xi)))))) / 2
            derivative += - (2 * v_xi.T.dot(dK_nm.dot(K_mm_inv.dot(B.dot(K_mm_inv.dot(K_mn.dot(v_xi)))))) -
                             2 * v_xi.T.dot(K_nm.dot(K_mm_inv.dot(dK_mm.dot(K_mm_inv.dot(B.dot(
                                                                        K_mm_inv.dot(K_mn.dot(v_xi)))))))) +
                             v_xi.T.dot(K_nm.dot(K_mm_inv.dot(dB.dot(K_mm_inv.dot(K_mn.dot(v_xi))))))
                             )
            derivative += np.trace(B_inv.dot(dB)) / 2
            if param != len(params) - 1:
                derivative += - np.trace(K_mnPsiK_nm_inv.dot(d_K_mnPsiK_nm)) / 2
            derivative += - np.sum(Psi_xi) * dK_nn
            derivative += (- np.trace(K_mm_inv.dot(dK_mm.dot(K_mm_inv.dot(K_mnPsiK_nm)))) +
                           np.trace(K_mm_inv.dot(d_K_mnPsiK_nm)))
            gradient.append(derivative[0, 0])
        return fun, np.array(gradient)
