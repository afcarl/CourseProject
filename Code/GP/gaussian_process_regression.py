import copy
import math
import time
from copy import deepcopy

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

from GP.covariance_functions import CovarianceFamily
from GP.gaussian_process import GP
from GP.gpr_res import GPRRes
from GP.optimization import gradient_descent, stochastic_gradient_descent, stochastic_average_gradient,\
                         minimize_wrapper, projected_newton, _eig_val_correction


class GPR(GP):
    """
    Gaussian Process Regressor
    """

    def __init__(self, cov_obj, mean_function=lambda x: 0, method='brute', parametrization='natural',
                 optimizer='L-BFGS-B'):
        """
        :param cov_obj: object of the CovarianceFamily class
        :param mean_function: function, mean of the gaussian process
        :param method: a string, showing, which method will be used for prediction and hyper-parameter optimization
            brute — full gaussian process regression
            vi — a method, using variational inference for finding inducing inputs
            means — a method, using k-means cluster centers as inducing inputs
            svi — a method, using stochastic variational inference for finding inducing inputs
        :param parametrization: the parametrization of the ELBO. Used only in the svi method, ignored otherwise.
            natural — natural gradient descent is used for optimization
            cholesky – cholesky decomposition is used for storing the covartiance matrix of the variational
            distribution
        :param optimizer: The optimization method, used to optimize the ELBO. Used only in the svi method,
        ignored otherwise.
            L-BFGS-B – the corresponding method from scipy
            SAG — Stochastic Average Gradient method
            FG — Gradient descent
        :return: GPR object
        """
        if not isinstance(cov_obj, CovarianceFamily):
            raise TypeError("The covariance object cov_obj is of the wrong type")
        if not hasattr(mean_function, '__call__'):
            raise TypeError("mean_function must be callable")
        if not method in ['brute', 'vi', 'means', 'svi']:
            raise ValueError("Invalid method name")
        if not parametrization in ['natural', 'cholesky']:
            raise ValueError("Invalid parametrization name")
        if not optimizer in ['SAG', 'L-BFGS-B', 'FG', 'Projected Newton', 'SG']:
            raise ValueError("Invalid optimizer name")

        self.covariance_fun = cov_obj.covariance_function
        self.covariance_obj = cov_obj
        self.mean_fun = mean_function
        self.method = method
        self.parametrization = parametrization
        self.optimizer = optimizer

        # A tuple: inducing inputs, and parameters of gaussian distribution at these points (mean and covariance)
        self.inducing_inputs = None

    # brute method

    @staticmethod
    def _ml(targets, cov_inv, cov_l):
        """
        :param targets: target_values list
        :param cov_inv: inversed covariance at data points
        :return: marginal likelihood
        """
        n = targets.size
        return -((targets.T.dot(cov_inv)).dot(targets) + 2 * np.sum(np.log(np.diag(cov_l))) +
                 n * np.log(2 * math.pi))/2

    @staticmethod
    def _ml_partial_derivative(targets, cov_inv, dk_dtheta_mat):
        """
        :param targets: target value vector
        :param cov_inv: inverse covariance matrix at data points
        :param dk_dtheta_mat: the matrix of partial derivatives of the covariance function at data points
        with respect to hyper-parameter theta
        :return: marginal likelihood partial derivative with respect to hyper-parameter theta
        """
        return ((((targets.T.dot(cov_inv)).dot(dk_dtheta_mat)).dot(cov_inv)).dot(targets) -
                np.trace(cov_inv.dot(dk_dtheta_mat))) / 2

    def _ml_grad(self, points, targets, cov_inv, params):
        """
        :param targets: target values' vector
        :param cov_inv: inverse covariance matrix
        :param points: data points array
        :return: marginal likelihood gradient with respect to hyperparameters
        """

        derivative_matrix_list = self.covariance_obj.get_derivative_function_list(params)
        # noise_derivative = 2 * params[-1] * np.eye(points.shape[1])
        noise_derivative = self.covariance_obj.get_noise_derivative(points.shape[1])
        return np.array([self._ml_partial_derivative(targets, cov_inv, func(points, points))
                         for func in derivative_matrix_list] +
                        [self._ml_partial_derivative(targets, cov_inv, noise_derivative)])

    def _oracle(self, points, targets, params):
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
        marginal_likelihood = self._ml(targets, points_cov_inv, points_l)
        gradient = self._ml_grad(points, targets, points_cov_inv, params)
        gradient = gradient.reshape(gradient.size, )
        return marginal_likelihood, gradient

    def _brute_fit(self, data_points, target_values, max_iter=None):
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
            loss, grad = self._oracle(data_points, target_values, w)
            return -loss, -grad

        bnds = self.covariance_obj.get_bounds()
        if max_iter is None:
            max_iter = np.inf
        res, w_list, time_list = minimize_wrapper(loc_fun, self.covariance_obj.get_params(), method='L-BFGS-B',
                                                               mydisp=False, bounds=bnds,
                                                               options={'gtol': 1e-8, 'ftol': 0, 'maxiter': max_iter})
        optimal_params = res.x
        self.covariance_obj.set_params(optimal_params)
        return GPRRes(deepcopy(w_list), time_lst=deepcopy(time_list))

    def _brute_predict(self, test_points, training_points, training_targets):
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

        return test_targets, low, up

    def _brute_get_prediction_quality(self, params, data_points, data_targets, test_points, test_targets):
        """
        Returns prediction quality on the test set for the given kernel (and inducing points) parameters for the means
        method
        :param params: parameters
        :param data_points: train set points
        :param data_targets: train set target values
        :param test_points: test set points
        :param test_targets: test set target values
        :return: prediction MSE
        """
        new_gp = deepcopy(self)
        new_gp.covariance_obj.set_params(params)
        predicted_y_test, _, _ = new_gp.predict(test_points, data_points, data_targets)
        return np.linalg.norm(predicted_y_test - test_targets)**2/test_targets.size

    # vi/means method

    def _vi_get_optimal_meancov(self, params, inputs, data_points, targets):

        cov_fun = self.covariance_obj.covariance_function
        sigma = params[-1]
        K_mm = cov_fun(inputs, inputs)
        K_mn = cov_fun(inputs, data_points)
        K_nm = K_mn.T

        Sigma = np.linalg.inv(K_mm + K_mn.dot(K_nm)/sigma**2)
        mu = sigma**(-2) * K_mm.dot(Sigma.dot(K_mn.dot(targets)))
        A = K_mm.dot(Sigma).dot(K_mm)
        return  mu, A
        # self.inducing_inputs = (inputs, mu, A)

    def _vi_means_fit(self, data_points, target_values, num_inputs, inputs=None, optimizer_options={}):
        """
        A procedure, fitting hyper-parameters and inducing points for both the 'means' and the 'vi' methods.
        :param data_points: data points
        :param target_values: target values at data points
        :param num_inputs: number of inducing inputs to be found
        :param max_iter: maximum number of iterations
        :return: lists of iteration-wise values of hyper-parameters, times, function values for evaluating the
        optimization
        """
        if not(isinstance(data_points, np.ndarray) and
               isinstance(target_values, np.ndarray)):
            raise TypeError("The operands must be of type numpy array")

        dim = data_points.shape[0]
        param_len = self.covariance_obj.get_params().size

        def _vi_loc_fun(w):
            ind_points = (w[param_len:]).reshape((dim, num_inputs)) # has to be rewritten for multidimensional case
            loss, grad = self._vi_means_oracle(data_points, target_values, w[:param_len], ind_points)
            return -loss, -grad

        def _means_loc_fun(w):
            loss, grad = self._vi_means_oracle(data_points, target_values, w, inputs)
            return -loss, -grad

        np.random.seed(15)
        if self.method == 'vi':
            inputs = data_points[:, :num_inputs] + np.random.normal(0, 0.1, (dim, num_inputs))
            loc_fun = _vi_loc_fun
            w0 = np.concatenate((self.covariance_obj.get_params(), inputs.ravel()))
            bnds = tuple(list(self.covariance_obj.get_bounds()) + [(1e-2, 1)] * num_inputs * dim)

        if self.method == 'means':
            if inputs is None:
                inputs = self._k_means_cluster_centers(data_points, num_inputs)
            loc_fun = _means_loc_fun
            w0 = self.covariance_obj.get_params()
            bnds = self.covariance_obj.get_bounds()

        if self.optimizer == 'L-BFGS-B':
            mydisp = False
            if not optimizer_options is None:
                if 'mydisp' in optimizer_options.keys():
                    mydisp = optimizer_options['mydisp']
                    del optimizer_options['mydisp']
            res, w_list, time_list = minimize_wrapper(loc_fun, w0, method='L-BFGS-B', mydisp=mydisp, bounds=bnds,
                                                      options=optimizer_options)
            res = res.x
        elif self.optimizer == 'Projected Newton':
            res, w_list, time_list = projected_newton(loc_fun, w0, bounds=bnds, options=optimizer_options)

        else:
            raise ValueError('Wrong optimizer for svi/means method:' + self.optimizer)

        if self.method == 'vi':
            optimal_params = res.x[:-num_inputs*dim]
            inducing_points = res.x[-num_inputs*dim:]
            inducing_points = inducing_points.reshape((dim, num_inputs))
        if self.method == 'means':
            # optimal_params = res.x
            optimal_params = res
            inducing_points = inputs
        self.covariance_obj.set_params(optimal_params)

        mu, Sigma = self._vi_get_optimal_meancov(optimal_params, inducing_points, data_points, target_values)
        self.inducing_inputs = (inducing_points, mu, Sigma)
        return GPRRes(deepcopy(w_list), time_lst=deepcopy(time_list))

    def _vi_means_oracle(self, points, targets, params, ind_points):
        """
        Oracle function for 'vi' and 'means' methods.
        :param points: data points array
        :param targets: target values vector
        :param params: hyper-parameters vector
        :param ind_points: inducing points
        """
        start = time.time()
        n = points.shape[1]
        m = ind_points.shape[1]
        sigma = params[-1]
        cov_obj = copy.deepcopy(self.covariance_obj)
        cov_obj.set_params(params)
        cov_fun = cov_obj.covariance_function
        K_mm = cov_fun(ind_points, ind_points)
        K_mm_l = np.linalg.cholesky(K_mm)
        K_mm_l_inv = np.linalg.inv(K_mm_l)
        K_mm_inv = K_mm_l_inv.T.dot(K_mm_l_inv)
        K_nm = cov_fun(points, ind_points)
        K_mn = K_nm.T
        K_mnK_nm = K_mn.dot(K_nm)
        Q_nn_tr = np.trace(K_mm_inv.dot(K_mnK_nm))
        try:
            anc_l = np.linalg.cholesky(K_mm + K_mnK_nm/sigma**2)
        except:
            # print(sigma)
            print('Warning, matrix is not positive definite', params)
            new_mat = _eig_val_correction(K_mm + K_mnK_nm/sigma**2, eps=10)
            # new_mat = (new_mat + new_mat.T)/2
            # new_mat += np.eye(m) * (np.abs(np.min(np.linalg.eigvals(new_mat))) + 1e-4)
            # print(np.linalg.eigvals(new_mat))
            anc_l = np.linalg.cholesky(new_mat)
            # raise ValueError('Singular matrix encountered. Parameters: ' + str(params))
        anc_l_inv = np.linalg.inv(anc_l)
        anc_inv = anc_l_inv.T.dot(anc_l_inv)
        K_mn_y = K_mn.dot(targets)
        y_B_inv_y = targets.T.dot(targets)/sigma**2 - K_mn_y.T.dot(anc_inv.dot(K_mn_y))/sigma**4
        B_inv_y = targets / sigma**2 - K_mn.T.dot(anc_inv.dot(K_mn.dot(targets)))/sigma**4
        B_log_det = (np.sum(np.log(np.diag(anc_l))) + n * np.log(sigma) - np.sum(np.log(np.diag(K_mm_l))))*2
        zero = np.array([[0]])
        K_nn_diag = cov_fun(zero, zero)
        F_v = - B_log_det/2 - y_B_inv_y/2 - \
              (K_nn_diag * n - Q_nn_tr) / (2 * sigma**2)

        # Gradient
        gradient = []

        derivative_matrix_list = cov_obj.get_derivative_function_list(params)
        A = anc_inv
        for func in derivative_matrix_list:
            dK_nm = func(points, ind_points)
            dK_mn = dK_nm.T
            dK_mm = func(ind_points, ind_points)
            dK_mm_inv = - K_mm_inv.dot(dK_mm.dot(K_mm_inv))
            K_mndK_nm = K_mn.dot(dK_nm)
            dB_dtheta_tr = 2 * np.trace(K_mm_inv.dot(K_mndK_nm)) + np.trace(dK_mm_inv.dot(K_mnK_nm))
            dB_B_inv_y = dK_nm.dot(K_mm_inv.dot(K_mn.dot(B_inv_y))) + K_nm.dot(dK_mm_inv.dot(K_mn.dot(B_inv_y)))\
                                 + K_nm.dot(K_mm_inv.dot(dK_mn.dot(B_inv_y)))
            y_B_inv_dB_B_inv_y = B_inv_y.T.dot(dB_B_inv_y)
            B_inv_dB_tr = dB_dtheta_tr / sigma**2 - \
                          (2 * np.trace((A.dot(K_mndK_nm)).dot(K_mm_inv.dot(K_mnK_nm)))
                           + np.trace((A.dot(K_mnK_nm)).dot(dK_mm_inv.dot(K_mnK_nm))))/sigma**4

            dK_nn = func(zero, zero)
            gradient.append((-B_inv_dB_tr / 2 + y_B_inv_dB_B_inv_y / 2 -
                             (dK_nn * n - dB_dtheta_tr) / (2 * sigma**2))[0, 0])

        # sigma derivative
        dK_mm = cov_obj.get_noise_derivative(K_mm.shape[0])
        dK_mm_inv = - K_mm_inv.dot(dK_mm.dot(K_mm_inv))

        dQ_dtheta_tr = np.trace(dK_mm_inv.dot(K_mnK_nm))
        dQ_B_inv_y = K_nm.dot(dK_mm_inv.dot(K_mn.dot(B_inv_y)))
        y_B_inv_dQ_B_inv_y = B_inv_y.T.dot(dQ_B_inv_y)
        y_B_inv_dB_B_inv_y = 2 * sigma * B_inv_y.T.dot(B_inv_y) + y_B_inv_dQ_B_inv_y
        dB_dtheta_tr = 2 * sigma * n + dQ_dtheta_tr
        B_inv_dB_tr = dB_dtheta_tr / sigma**2 - \
                      (np.trace(A.dot(K_mnK_nm).dot(dK_mm_inv).dot(K_mnK_nm))
                       + 2 * sigma * np.trace(A.dot(K_mnK_nm)))/sigma**4
        dK_nn = 2 * sigma
        gradient.append((- B_inv_dB_tr / 2 + y_B_inv_dB_B_inv_y / 2 -
                (n * dK_nn - dQ_dtheta_tr) / (2 * sigma**2))[0, 0] + (n * K_nn_diag[0, 0] - Q_nn_tr) / sigma**3)

        # inducing points derivatives
        # By now this is not written in an optimal way
        if self.method == 'vi':
            B_inv = np.eye(n)/sigma**2 - K_mn.T.dot(anc_inv.dot(K_mn))/sigma**4
            # print('vi method might not work propperly in the current version')
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
                    gradient.append(self._vi_lower_bound_partial_derivative(targets, dB_dtheta, dB_dtheta, B_inv, sigma,
                                                                             dK_nn))
        return F_v[0, 0], np.array(gradient)

    def _vi_lower_bound_partial_derivative(self, y, dB_dtheta_mat, dQ_dtheta_mat, B_inv, sigma, dK_nn_dtheta):
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

    def _means_get_prediction_quality(self, params, data_points, data_targets, test_points, test_targets):
        """
        Returns prediction quality on the test set for the given kernel (and inducing points) parameters for the means
        method
        :param params: parameters
        :param data_points: train set points
        :param data_targets: train set target values
        :param test_points: test set points
        :param test_targets: test set target values
        :return: prediction MSE
        """
        new_gp = deepcopy(self)
        new_gp.covariance_obj.set_params(params)
        mu, Sigma = new_gp._vi_get_optimal_meancov(params, new_gp.inducing_inputs[0], data_points, data_targets)
        new_gp.inducing_inputs = (new_gp.inducing_inputs[0], mu, Sigma)
        predicted_y_test, _, _ = new_gp.predict(test_points)
        return r2_score(test_targets, predicted_y_test)
        # return np.linalg.norm(predicted_y_test - test_targets)**2/test_targets.size

    @staticmethod
    def _k_means_cluster_centers(data_points, num_clusters):
        """
        K-Means clusterization for data points
        :param data_points: data points
        :param num_inputs: number of clusters
        :return: K-Means cluster centers
        """
        means = KMeans(n_clusters=num_clusters)
        means.fit(data_points.T)
        return means.cluster_centers_.T

    # svi method

    @staticmethod
    def _svi_lower_triang_mat_to_vec(mat):
        """
        Transforms a lower-triangular matrix to a vector of it's components, that are lower than the main diagonal
        :param mat: lower-triangular matrix
        :return: a vector
        """
        indices = np.tril_indices(mat.shape[0])
        vec = mat[indices]
        return vec

    @staticmethod
    def _svi_lower_triang_vec_to_mat(vec):
        """
        Transforms a vector similar to the ones, produced by _svi_lower_triang_mat_to_vec, to a lower-diagonal matrix
        :param vec: a vector of the lower-triangular matrix' components, that are lower than the main diagonal
        :return: a lower-triangular matrix
        """
        m = len(vec)
        k = (-1 + np.sqrt(1 + 8 * m)) / 2
        if k != int(k):
            raise ValueError("Vec has an invalid size")
        indices = np.tril_indices(k)
        mat = np.zeros((k, k))
        mat[indices] = vec.reshape(-1)
        return mat

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
        if self.parametrization == 'cholesky':
            eta_2 = self._svi_lower_triang_mat_to_vec(eta_2)[:, None]
        else:
            eta_2 = eta_2.reshape(-1)[:, None]
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
        if self.parametrization == 'natural':
            mat_size = np.int(np.sqrt((vec.size - theta_len + 1/4)) - 1 / 2)
        elif self.parametrization == 'cholesky':
            mat_size = np.int(np.sqrt(2 * (vec.size - theta_len) + 9 / 4) - 3 / 2)
        eta_1 = vec[theta_len:theta_len+mat_size, :].reshape((mat_size, 1))
        if self.parametrization == 'cholesky':
            eta_2 = self._svi_lower_triang_vec_to_mat(vec[theta_len+mat_size:, :])
        else:
            eta_2 = vec[theta_len+mat_size:, :].reshape((mat_size, mat_size))
        return theta, eta_1, eta_2

    def _svi_get_bounds(self, m):
        bnds = list(self.covariance_obj.get_bounds())
        if self.parametrization == 'natural':
            return tuple(bnds +
                         [(None, None)] * (m + m*m))
        elif self.parametrization == 'cholesky':
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
        mu = np.zeros((m, 1))
        sigma_n = self.covariance_obj.get_params()[-1]

        theta = self.covariance_obj.get_params()
        if self.parametrization == 'natural':
            # sigma_inv = np.eye(m)

            ######################################################
            # Experimental
            cov_fun = self.covariance_obj.covariance_function
            K_mn = cov_fun(inputs, data_points)
            K_mm = cov_fun(inputs, inputs)
            K_mm_inv = np.linalg.inv(K_mm)
            sigma_inv = K_mm_inv.dot(K_mn.dot(K_mn.T.dot(K_mm_inv)))/sigma_n**2 + K_mm_inv
            sigma = np.linalg.inv(sigma_inv)
            mu = sigma.dot(K_mm_inv.dot((K_mn.dot(y)))) / sigma_n**2
            #######################################################

            # Canonical parameters initialization
            eta_1 = sigma_inv.dot(mu)
            eta_2 = - sigma_inv / 2
            param_vec = self._svi_get_parameter_vector(theta, eta_1, eta_2)

        elif self.parametrization == 'cholesky':
            # sigma_L = np.eye(m)  # Cholesky factor of sigma

            ######################################################
            # Experimental
            cov_fun = self.covariance_obj.covariance_function
            K_mn = cov_fun(inputs, data_points)
            K_mm = cov_fun(inputs, inputs)
            K_mm_inv = np.linalg.inv(K_mm)
            sigma = np.linalg.inv(K_mm_inv.dot(K_mn.dot(K_mn.T.dot(K_mm_inv)))/sigma_n**2 + K_mm_inv)
            mu = sigma.dot(K_mm_inv.dot((K_mn.dot(y)))) / sigma_n**2
            #######################################################

            # p = np.random.normal(size=(m, 1))
            # sigma = p.dot(p.T) + np.eye(m) * 1e-4
            sigma_L = np.linalg.cholesky(sigma)
            param_vec = self._svi_get_parameter_vector(theta, mu, sigma_L)

        bnds = self._svi_get_bounds(m)

        if self.parametrization == 'natural':

            def stoch_fun(x, i):
                return -self._svi_elbo_batch_approx_oracle(data_points, target_values, inputs, parameter_vec=x,
                                                           indices=i)[1]

            res, w_list, time_list = stochastic_gradient_descent(oracle=stoch_fun, n=n, point=param_vec, bounds=bnds,
                                                                 options=optimizer_options)

            theta, eta_1, eta_2 = self._svi_get_parameters(res)
            sigma_inv = - 2 * eta_2
            sigma = np.linalg.inv(sigma_inv)
            mu = sigma.dot(eta_1)

        elif self.parametrization == 'cholesky':
            def fun(x):
                fun, grad = self._svi_elbo_batch_approx_oracle(data_points, target_values, inputs, parameter_vec=x,
                                                     indices=list(range(n)))
                return -fun, -grad

            def sag_oracle(x, i):
                fun, grad = self._svi_elbo_batch_approx_oracle(data_points, target_values, inputs, parameter_vec=x,
                                                               indices=i)
                return -fun, -grad

            if self.optimizer == 'SAG':
                res, w_list, time_list = stochastic_average_gradient(oracle=sag_oracle, n=n, point=param_vec,
                                                                     bounds=bnds, options=optimizer_options)
            elif self.optimizer == 'FG':
                res, w_list, time_list = gradient_descent(oracle=fun, point=param_vec, bounds=bnds,
                                                          options=optimizer_options)
            elif self.optimizer == 'L-BFGS-B':
                res, w_list, time_list = minimize_wrapper(fun, param_vec, method='L-BFGS-B', mydisp=False,
                                                          bounds=bnds, jac=True, options=optimizer_options)
                res = res['x']
            else:
                raise ValueError('Wrong optimizer for svi method' + self.optimizer)

            theta, mu, sigma_L = self._svi_get_parameters(res)
            sigma = sigma_L.dot(sigma_L.T)

        self.covariance_obj.set_params(theta)
        self.inducing_inputs = (inputs, mu, sigma)
        return GPRRes(deepcopy(w_list), time_lst=deepcopy(time_list))

    def _svi_get_loss(self, params, data_points, target_values):
        new_gp = deepcopy(self)
        n = target_values.size
        def fun(x):
            fun, grad = new_gp._svi_elbo_batch_approx_oracle(data_points, target_values, new_gp.inducing_inputs[0],
                                                             parameter_vec=x, indices=list(range(n)))
            return -fun, -grad
        return fun(params)[0][0,0]


    def _svi_elbo_batch_approx_oracle(self, data_points, target_values, inducing_inputs, parameter_vec,
                                       indices):
        """
        The approximation of Evidence Lower Bound (L3 from the article 'Gaussian process for big data') and it's derivative wrt
        kernel hyper-parameters. The approximation is found using just one data point.
        :param data_points: the array of data points
        :param target_values: the target values at these points
        :param inducing_inputs: an array of inducing inputs
        :param mu: the current mean of the process at the inducing points
        :param sigma: the current covariance of the process at the inducing points
        :param theta: a vector of hyper-parameters, the point of evaluation
        :param indices: a list of indices of the data points in the mini-batch
        :return: ELBO and it's gradient approximation in a tuple
        """

        N = target_values.size
        m = inducing_inputs.shape[1]

        if self.parametrization == 'natural':
            theta, eta_1, eta_2 = self._svi_get_parameters(parameter_vec)
            sigma_inv = - 2 * eta_2
            sigma = np.linalg.inv(sigma_inv)
            sigma_L = np.linalg.cholesky(sigma)
            mu = sigma.dot(eta_1)
        elif self.parametrization == 'cholesky':
            theta, mu, sigma_L = self._svi_get_parameters(parameter_vec)
            sigma = sigma_L.dot(sigma_L.T)

        old_params = self.covariance_obj.get_params()
        # theta = list(theta)
        # theta.append(np.copy(old_params[-1]))
        # theta = np.array(theta)
        self.covariance_obj.set_params(theta)

        # Data point for gradient estimation
        # if indices is None:
        #     index = np.random.randint(0, high=N-1)

        l = len(indices)
        i = indices
        y_i = target_values[i]
        x_i = data_points[:, i]

        # Covariance function and it's parameters
        cov_fun = self.covariance_fun
        params = self.covariance_obj.get_params()
        # sigma_n = self.covariance_obj.get_params()[-1]
        sigma_n = parameter_vec[len(self.covariance_obj.get_params())-1]

        # Covariance matrices
        K_mm = cov_fun(inducing_inputs, inducing_inputs)
        L = np.linalg.cholesky(K_mm)
        L_inv = np.linalg.inv(L)
        K_mm_inv = L_inv.T.dot(L_inv)
        k_i = cov_fun(inducing_inputs, x_i)
        Lambda_i = K_mm_inv.dot(k_i.dot(k_i.T.dot(K_mm_inv))) / sigma_n**2
        tilde_K_ii = l * cov_fun(x_i[:, :1], x_i[:, :1]) - np.einsum('ij,ji->', k_i.T, K_mm_inv.dot(k_i))

        # Derivatives
        derivative_matrix_list = self.covariance_obj.get_derivative_function_list(params)
        d_K_mm__d_theta_lst = [fun(inducing_inputs, inducing_inputs) for fun in derivative_matrix_list]
        d_k_i__d_theta_lst = [fun(inducing_inputs, data_points[:, i]) for fun in derivative_matrix_list]
        d_K_mm__d_sigma_n = self.covariance_obj.get_noise_derivative(points_num=m)
        d_k_i__d_sigma_n = np.zeros((m, l))
        d_K_mm__d_theta_lst.append(d_K_mm__d_sigma_n)
        d_k_i__d_theta_lst.append(d_k_i__d_sigma_n)

        # Variational Lower Bound, estimated by one data point
        loss = -np.log(sigma_n) * l - np.linalg.norm(y_i - k_i.T.dot(K_mm_inv.dot(mu)))**2 / (2 * sigma_n**2)
        loss += - tilde_K_ii / (2 * sigma_n**2)
        loss += - np.trace(sigma.dot(Lambda_i))/2
        loss += - np.sum(np.log(np.diag(L))) * l / N
        loss += np.sum(np.log(np.diag(sigma_L))) * l / N
        loss += - np.trace(sigma.dot(K_mm_inv)) * l / (2*N)
        loss += - mu.T.dot(K_mm_inv.dot(mu)) * l / (2*N)

        # Gradient
        grad = np.zeros((len(theta,)))
        for param in range(len(theta)):
            if param != len(theta) - 1:
                cov_derivative = derivative_matrix_list[param]
            else:
                cov_derivative = lambda x, y: self.covariance_obj.get_noise_derivative(points_num=1)

            d_K_mm__d_theta = d_K_mm__d_theta_lst[param]
            d_k_i__d_theta = d_k_i__d_theta_lst[param]
            grad[param] += np.sum((y_i - k_i.T.dot(K_mm_inv.dot(mu))) * \
                             (d_k_i__d_theta.T.dot(K_mm_inv) -
                              k_i.T.dot(K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv)))).dot(mu) / sigma_n**2)
            grad[param] += (2 * np.einsum('ij,ji->', d_k_i__d_theta.T, K_mm_inv.dot(k_i)) -
                            np.einsum('ij,ji->', k_i.T, K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv.dot(k_i))))
                            ) / (2 * sigma_n**2) - cov_derivative(x_i[:, :1], x_i[:, :1]) * l / (2 * sigma_n**2)
            grad[param] += np.trace(sigma.dot(
                K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv.dot(k_i.dot(k_i.T.dot(K_mm_inv))))) -
                K_mm_inv.dot(d_k_i__d_theta.dot(k_i.T.dot(K_mm_inv)))
            )) / sigma_n**2
            grad[param] += - np.trace(K_mm_inv.dot(d_K_mm__d_theta)) * l / (2*N)
            grad[param] += np.trace(sigma.dot(K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv)))) * l / (2*N)
            grad[param] += mu.T.dot(K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv.dot(mu)))) * l / (2*N)

        # print(l)
        # print(sigma_n)
        # exit(0)
        grad[-1] += (
            tilde_K_ii / (sigma_n**3) -
            l / (sigma_n) +
            np.linalg.norm(y_i - k_i.T.dot(K_mm_inv.dot(mu)))**2 / (sigma_n**3) +
            np.trace(sigma.dot(Lambda_i)) / sigma_n)

        grad = grad[:, None]

        if self.parametrization == 'natural':

            dL_dbeta1 = - (K_mm_inv.dot(k_i)).dot(y_i) / sigma_n**2 + eta_1 * l / N
            dL_dbeta2 = ((-Lambda_i - K_mm_inv * l / N) / 2 - eta_2 * l / N) * 2 # Don't quite get this *2
            grad = np.vstack((grad, -dL_dbeta1.reshape(-1)[:, None]))
            grad = np.vstack((grad, dL_dbeta2.reshape(-1)[:, None]))

        elif self.parametrization == 'cholesky':
            dL_dsigma_L = -Lambda_i.dot(sigma_L) + np.eye(m) * l / (N * np.diag(sigma_L)) - (K_mm_inv.dot(sigma_L)) * l / N
            dL_dsigma_L = self._svi_lower_triang_mat_to_vec(dL_dsigma_L)
            dL_dmu = -(K_mm_inv.dot(k_i)).dot(y_i) /sigma_n**2 + (Lambda_i + K_mm_inv *l/N).dot(mu)
            grad = np.vstack((grad, -dL_dmu.reshape(-1)[:, None]))
            grad = np.vstack((grad, dL_dsigma_L.reshape(-1)[:, None]))

        self.covariance_obj.set_params(old_params)

        return loss, grad[:, 0]

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
        # param_len = len(new_gp.covariance_obj.get_params())
        if self.parametrization == 'natural':
            theta, eta_1, eta_2 = new_gp._svi_get_parameters(params)
            Sigma_inv = - 2 * eta_2
            Sigma = np.linalg.inv(Sigma_inv)
            mu = Sigma.dot(eta_1)
        elif self.parametrization == 'cholesky':
            theta, mu, Sigma_L = new_gp._svi_get_parameters(params)
            Sigma = Sigma_L.dot(Sigma_L.T)
        theta = params[:len(new_gp.covariance_obj.get_params())]
        new_gp.covariance_obj.set_params(theta)
        new_gp.inducing_inputs = (new_gp.inducing_inputs[0], mu, Sigma)
        predicted_y_test, _, _ = new_gp.predict(test_points)
        return r2_score(test_targets, predicted_y_test)
        # return np.linalg.norm(predicted_y_test - test_targets)**2/test_targets.size

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
        return test_targets, up, low

    # General functions

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
        targets = targets.reshape((targets.size, 1))
        return targets[:tr_points.shape[1], :], targets[tr_points.shape[1]:, :]

    def fit(self, *args, **kwargs):
        if self.method == 'brute':
            return self._brute_fit(*args, **kwargs)
        if self.method == 'vi' or self.method == 'means':
            return self._vi_means_fit(*args, **kwargs)
        if self.method == 'svi':
            return self._svi_fit(*args, **kwargs)
        else:
            print(self.method)
            raise ValueError("Method " + self.method + " is invalid")

    def predict(self, *args, **kwargs):
        if self.method == 'brute':
            return self._brute_predict(*args, **kwargs)
        else:
            return self._inducing_points_predict(*args, **kwargs)

    def get_prediction_quality(self, *args, **kwargs):
        """
        Returns prediction quality on the test set for the given kernel (and inducing points) parameters
        :param params: parameters
        :return: prediction r2
        """
        if self.method == 'means':
            return self._means_get_prediction_quality(*args, **kwargs)
        elif self.method == 'svi':
            return self._svi_get_prediction_quality(*args, **kwargs)
        elif self.method == 'brute':
            return self._brute_get_prediction_quality(*args, **kwargs)
        else:
            raise ValueError('Wrong method')

    def get_loss(self, *args, **kwargs):
        """
        Returns loss (ELBO) kernel (and inducing points) parameters
        :return: loss
        """
        if self.method == 'svi':
            return self._svi_get_loss(*args, **kwargs)
        else:
            raise ValueError('Wrong method')

