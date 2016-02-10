import numpy as np
import math
import copy
from sklearn.cluster import KMeans

from gaussian_process import GP, minimize_wrapper
from covariance_functions import CovarianceFamily, sigmoid


class GPR(GP):
    """
    Gaussian Process Regressor
    """

    def __init__(self, cov_obj, mean_function=lambda x: 0, method='brute'):
        """
        :param cov_obj: object of the CovarianceFamily class
        :param mean_function: function, mean of the gaussian process
        :param method: a string, showing, which method will be used for prediction and hyper-parameter optimization
        brute — full gaussian process regression
        vi — a method, using variational inference for finding inducing inputs
        means — a method, using k-means cluster centers as inducing inputs
        svi — a method, using stochastic variational inference for finding inducing inputs
        :return: GPR object
        """
        if not isinstance(cov_obj, CovarianceFamily):
            raise TypeError("The covariance object cov_obj is of the wrong type")
        if not hasattr(mean_function, '__call__'):
            raise TypeError("mean_function must be callable")
        if not method in ['brute', 'vi', 'means', 'svi']:
            raise ValueError("Invalid method name")

        self.covariance_fun = cov_obj.covariance_function
        self.covariance_obj = cov_obj
        self.mean_fun = mean_function
        self.method = method
        # A tuple: inducing inputs, and parameters of gaussian distribution at these points (means and covariances)
        self.inducing_inputs = None

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
                                                           mydisp=False, bounds=bnds, options={'gtol': 1e-8, 'ftol': 0,
                                                                                     'maxiter': max_iter})
        optimal_params = res.x
        self.covariance_obj.set_params(optimal_params)
        return w_list, time_list, fun_lst

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
        # if test_points.shape[0] == 1:
        #     gp_plot_reg_data(test_points, up, 'r')
        #     gp_plot_reg_data(test_points, low, 'g')
        #     gp_plot_reg_data(test_points, test_targets, 'b')
        return test_targets, low, up

    def fit(self, *args, **kwargs):
        if self.method == 'brute':
            return self._brute_fit(*args, **kwargs)
        if self.method == 'vi' or self.method == 'means':
            return self._vi_means_fit(*args, **kwargs)
        else:
            print(self.method)
            raise ValueError("Method" + self.method + " is invalid")

    def predict(self, *args, **kwargs):
        if self.method == 'brute':
            return self._brute_predict(*args, **kwargs)
        else:
            return self._inducing_points_predict(*args, **kwargs)

    def _vi_means_fit(self, data_points, target_values, num_inputs, max_iter=None):
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

        if max_iter is None:
            max_iter = np.inf

        def _vi_loc_fun(w):
                # print("Params:", w[:param_len])

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
            inputs = self._k_means_cluster_centers(data_points, num_inputs)
            loc_fun = _means_loc_fun
            w0 = self.covariance_obj.get_params()
            bnds = self.covariance_obj.get_bounds()

        # f2, g2 = loc_fun(w0)
        # print("Gradient:", g2)
        # for i in range(w0.size):
        #     diff = np.zeros(w0.shape)
        #     diff[i] = 1e-8
        #     f1, g1 = loc_fun(w0 + diff)
        #     print("Difference:", (f1 - f2) * 1e8)
        # exit(0)
        res, w_list, time_list, fun_lst = minimize_wrapper(loc_fun, w0, method='L-BFGS-B',
                                                  mydisp=False, bounds=bnds, options={'gtol': 1e-8, 'ftol': 0,
                                                                                     'maxiter': max_iter})
        if self.method == 'vi':
            optimal_params = res.x[:-num_inputs*dim]
            inducing_points = res.x[-num_inputs*dim:]
            inducing_points = inducing_points.reshape((dim, num_inputs))
        if self.method == 'means':
            optimal_params = res.x
            inducing_points = inputs
        self.covariance_obj.set_params(optimal_params)

        cov_fun = self.covariance_obj.covariance_function
        sigma = optimal_params[-1]
        K_mm = cov_fun(inducing_points, inducing_points)
        K_mn = cov_fun(inducing_points, data_points)
        K_nm = K_mn.T
        Sigma = np.linalg.inv(K_mm + K_mn.dot(K_nm)/sigma**2)
        mu = sigma**(-2) * K_mm.dot(Sigma).dot(K_mn).dot(target_values)
        A = K_mm.dot(Sigma).dot(K_mm)
        self.inducing_inputs = (inducing_points, mu, A)
        return w_list, time_list, fun_lst

    def _vi_means_oracle(self, points, targets, params, ind_points):
        """
        Oracle function for 'vi' and 'means' methods.
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
            gradient.append(self._lower_bound_partial_derivative(targets, dB_dtheta, dB_dtheta, B_inv, sigma,
                                                                         dK_nn))

        # sigma derivative
        dK_mm = cov_obj.get_noise_derivative(K_mm.shape[0])
        dK_mm_inv = - K_mm_inv.dot(dK_mm.dot(K_mm_inv))
        dQ_dtheta = K_nm.dot(dK_mm_inv).dot(K_mn)
        dB_dtheta = dQ_dtheta + 2 * sigma * np.eye(Q_nn.shape[0])
        # dK_nn = (zero[:, :, None], zero[:, None, :])
        dK_nn = 2 * sigma
        gradient.append(self._lower_bound_partial_derivative(targets, dB_dtheta, dQ_dtheta, B_inv, sigma,
                                                                     dK_nn))
        gradient[-1] += np.sum(K_nn_diag - np.diag(Q_nn)) / sigma**3

        # inducing points derivatives
        if self.method == 'vi':
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
                    gradient.append(self._lower_bound_partial_derivative(targets, dB_dtheta, dB_dtheta, B_inv, sigma,
                                                                             dK_nn))
        return F_v[0, 0], np.array(gradient)

    def _lower_bound_partial_derivative(self, y, dB_dtheta_mat, dQ_dtheta_mat, B_inv, sigma, dK_nn_dtheta):
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


    def _svi_find_hyper_parameters(self, data_points, target_values, inputs=None, num_inputs=0, max_iter=None):
        """
        A method for optimizing hyper-parameters (for fixed inducing points), based on stochastic variational inference
        :param data_points:
        :param target_values:
        :param inputs:
        :param num_inputs:
        :param max_iter:
        :return:
        """

        # if no inducing inputs are provided, we use K-Means cluster centers as inducing inputs
        if inputs is None:
            means = KMeans(n_clusters=num_inputs)
            means.fit(data_points.T)
            inputs = means.cluster_centers_.T



