import copy
import time

import numpy as np
import scipy as sp
import scipy.optimize as op
from sklearn.cluster import KMeans
from numpy.polynomial.hermite import hermgauss

from GP.covariance_functions import CovarianceFamily, sigmoid
from GP.gaussian_process import GP
from GP.optimization import check_gradient, minimize_wrapper


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
            'svi' — inducing input method
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

    def _brute_fit(self, points, labels, max_iter=10):
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
        elif self.method == 'svi':
            # raise ValueError("Not Implemented yet")
            return self._inducing_points_predict(*args, **kwargs)
            # return self._inducing_points_predict(*args, **kwargs)
        else:
            raise ValueError("Unknown method")

    def fit(self, *args, **kwargs):
        if self.method == 'brute':
            return self._brute_fit(*args, **kwargs)
        elif self.method == 'brute_alt':
            return self._brute_alternative_fit(*args, **kwargs)
        elif self.method == 'svi':
            return self._svi_fit(*args, **kwargs)
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
        :param max_iter: maximum number of iterations in stochastic gradient descent
        :return:
        """

        # if no inducing inputs are provided, we use K-Means cluster centers as inducing inputs
        if inputs is None:
            means = KMeans(n_clusters=num_inputs)
            means.fit(data_points.T)
            inputs = means.cluster_centers_.T
            # inputs = np.load("inputs.npy")

        # Initializing required variables
        y = target_values
        m = num_inputs
        n = y.size


        # Initializing variational (normal) distribution parameters
        # mu = np.zeros((m, 1))
        mu = np.array([1, 2, 3])[:, None]
        sigma_L = np.eye(m)  # Cholesky factor of sigma
        # sigma_L = np.array([[1, 0, 0], [1, 2, 0], [1, 2, 3]])
        sigma = sigma_L.dot(sigma_L.T)

        theta = self.covariance_obj.get_params()
        param_vec = self._svi_get_parameter_vector(theta, mu, sigma_L)

        bnds = self._svi_get_bounds(m)

        def fun(w):
            # print(w.shape)
            fun, grad = self._svi_elbo_batch_approx_oracle(data_points, target_values, inputs, parameter_vec=w,
                                       indices=range(target_values.size))
            return -fun, -grad[:, 0]
        mydisp = False
        if not optimizer_options is None:
            if 'mydisp' in optimizer_options.keys():
                mydisp = optimizer_options['mydisp']
                del options['mydisp']
        res, w_list, time_list = minimize_wrapper(fun, param_vec, method='L-BFGS-B', mydisp=mydisp,
                                                          bounds=bnds, jac=True, options=optimizer_options)
        # print(res)
        # exit(0)
        res = res['x']
        theta, mu, sigma_L = self._svi_get_parameters(res)
        sigma = sigma_L.dot(sigma_L.T)

        self.inducing_inputs = (inputs, mu, sigma)


    def _svi_elbo_batch_approx_oracle(self, data_points, target_values, inducing_inputs, parameter_vec,
                                       indices):
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
        L = np.linalg.cholesky(K_mm)
        L_inv = np.linalg.inv(L)
        K_mm_inv = L_inv.T.dot(L_inv)
        k_i = cov_fun(inducing_inputs, x_i)
        Lambda_i = K_mm_inv.dot(k_i.dot(k_i.T.dot(K_mm_inv))) / sigma_n**2
        K_ii = cov_fun(x_i[:, :1], x_i[:, :1])
        tilde_K_ii = l * K_ii - np.einsum('ij,ji->', k_i.T, K_mm_inv.dot(k_i))

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
        mat = np.exp(mat)
        mat += 1
        mat = -np.log(mat)
        mat *= weights
        return np.sum(mat) / np.sqrt(np.pi)

    def _svi_compute_mu_grad_expectation(self, means, stds, targets):
        points, weights = self.gauss_hermite
        points = points[None, :]
        weights = weights[None, :]
        mat = targets / (1 + np.exp(targets * (np.sqrt(2) * points * stds + means)))
        mat *= weights
        return (np.sum(mat, axis=1) / np.sqrt(np.pi))[None, :]

    def _svi_compute_sigma_l_grad_expectation(self, means, stds, targets):
        points, weights = self.gauss_hermite
        points = points[None, :]
        weights = weights[None, :]
        anc_mat = np.exp(targets * (np.sqrt(2) * points * stds + means))
        mat = -targets**2 * anc_mat / (1 + anc_mat)**2
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

