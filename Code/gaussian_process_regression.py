import numpy as np
from scipy.optimize import minimize
import math
import copy
from sklearn.cluster import KMeans

from gaussian_process import GP, minimize_wrapper
from covariance_functions import CovarianceFamily, sigmoid
from optimization import gradient_descent


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

        # A tuple: inducing inputs, and parameters of gaussian distribution at these points (mean and covariance)
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

    def _svi_get_parameter_vector(self, theta, eta_1, eta_2):

        theta = np.array(theta).reshape(-1)[:, None]
        eta_1 = eta_1.reshape(-1)[:, None]
        eta_2 = eta_2.reshape(-1)[:, None]
        # print(eta_1.shape)
        return np.vstack((theta, eta_1, eta_2))[:, 0]

    def _svi_get_parameters(self, vec):
        theta_len = len(self.covariance_obj.get_params())
        vec = vec[:, None]
        theta = vec[:theta_len, :]
        mat_size = np.int(np.sqrt((vec.size - theta_len+ 1/4)) - 1 / 2)
        eta_1 = vec[theta_len:theta_len+mat_size, :].reshape((mat_size, 1))
        eta_2 = vec[theta_len+mat_size:, :].reshape((mat_size, mat_size))
        return theta, eta_1, eta_2



    def _svi_fit(self, data_points, target_values, num_inputs=0, inputs=None, max_iter=1):
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

        # Initializing required variables
        y = target_values
        m = num_inputs
        n = y.size

        # Initializing variational (normal) distribution parameters
        # mu = np.load("mu.npy")[:num_inputs, :]
        # sigma = np.load("sigma.npy")[:num_inputs, :num_inputs]

        mu = np.zeros((m, 1))
        sigma = np.eye(m)
        # sigma_inv = np.linalg.inv(sigma)

        # # Learning rate
        # alpha = 1e-2
        # gamma = 0.55

        # # Canonical parameters initialization
        # eta_1 = sigma_inv.dot(mu)
        # eta_2 = - sigma_inv / 2

        # Expectation parameters
        beta_1 = mu
        beta_2 = mu.dot(mu.T) + sigma

        #ELBO check

        theta = self.covariance_obj.get_params()
        param_vec = self._svi_get_parameter_vector(theta, beta_1, beta_2)

        # # Debug
        # sigma_n = self.covariance_obj.get_params()[-1]
        # cov_fun = self.covariance_obj.covariance_function
        # K_mm = cov_fun(inputs, inputs)
        # K_mn = cov_fun(inputs, data_points)
        # L = np.linalg.cholesky(K_mm)
        # L_inv = np.linalg.inv(L)
        # K_mm_inv = L_inv.T.dot(L_inv)
        # Lambda = K_mm_inv.dot(K_mn.dot(K_mn.T.dot(K_mm_inv))) / sigma_n + K_mm_inv
        # sigma_inv = np.linalg.inv(sigma)

        def fun(x):
            full_loss = 0
            full_grad = np.zeros((len(param_vec),))
            for i in range(n):
                full_loss += self._svi_elbo_approx_oracle(data_points, target_values, inputs, parameter_vec=x, index=i)[0]
                full_grad += self._svi_elbo_approx_oracle(data_points, target_values, inputs, parameter_vec=x, index=i)[1]
            # print(full_grad)
            # exit(0)
            return -full_loss, -np.array(full_grad)

        bnds = tuple(list(self.covariance_obj.get_bounds()) + [(None, None)] * (len(param_vec) - len(theta)))

        res = gradient_descent(oracle=fun, point=param_vec, bounds=bnds, options={'maxiter': 100, 'verbose': True,
                                                                                  'print_freq':5})
        theta, beta_1, beta_2 = self._svi_get_parameters(res)

        # res = minimize(fun, param_vec, method='L-BFGS-B', jac=True, bounds=bnds, options={'maxiter': 30, 'iprint':True, 'disp':True})
        # theta, beta_1, beta_2 = self._svi_get_parameters(res['x'])
        # self.covariance_obj.set_params(theta)
        # # np.save("theta.npy", theta)

        sigma = beta_2 - beta_1.dot(beta_1.T)
        # sigma_inv = np.linalg.inv(sigma)
        mu = beta_1
        print(mu)

        self.inducing_inputs = (inputs, mu, sigma)

    def _svi_elbo_approx_oracle(self, data_points, target_values, inducing_inputs, parameter_vec,
                                       index=None):
        """
        The approximation of Evidence Lower Bound (L3 from the article 'Gaussian process for big data') and it's derivative wrt
        kernel hyper-parameters. The approximation is found using just one data point.
        :param data_points: the array of data points
        :param target_values: the target values at these points
        :param inducing_inputs: an array of inducing inputs
        :param mu: the current mean of the process at the inducing points
        :param sigma: the current covariance of the process at the inducing points
        :param theta: a vector of hyper-parameters, the point of evaluation
        :param index: the number of the datapoint to be considered
        :return: ELBO and it's gradient approximation in a tuple
        """
        # theta, eta_1, eta_2 = self._svi_get_parameters(parameter_vec)
        # sigma_inv = - 2 * eta_2
        # sigma = np.linalg.inv(sigma_inv)
        # mu = sigma.dot(eta_1)

        theta, beta_1, beta_2 = self._svi_get_parameters(parameter_vec)
        sigma = beta_2 - beta_1.dot(beta_1.T)
        sigma_inv = np.linalg.inv(sigma)
        mu = beta_1
        eta_1 = sigma_inv.dot(mu)
        eta_2 = - sigma_inv / 2

        self.covariance_obj.set_params(theta)
        old_params = self.covariance_obj.get_params()

        N = target_values.size
        m = inducing_inputs.shape[1]
        # Data point for gradient estimation
        if index is None:
            index = np.random.randint(0, high=N-1)

        i = index
        y_i = target_values[i]
        x_i = data_points[:, i][:, None]

        # Covariance function and it's parameters
        cov_fun = self.covariance_fun
        params = self.covariance_obj.get_params()
        sigma_n = self.covariance_obj.get_params()[-1]
        # print(sigma_n)

        # Covariance matrices
        K_mm = cov_fun(inducing_inputs, inducing_inputs)
        L = np.linalg.cholesky(K_mm)
        L_inv = np.linalg.inv(L)
        K_mm_inv = L_inv.T.dot(L_inv)
        k_i = cov_fun(inducing_inputs, data_points[:, i][:, None])
        Lambda_i = K_mm_inv.dot(k_i.dot(k_i.T.dot(K_mm_inv))) / sigma_n
        tilde_K_ii = cov_fun(x_i, x_i) - k_i.T.dot(K_mm_inv.dot(k_i))

        # Derivatives
        derivative_matrix_list = self.covariance_obj.get_derivative_function_list(params)
        d_K_mm__d_theta_lst = [fun(inducing_inputs, inducing_inputs) for fun in derivative_matrix_list]
        d_k_i__d_theta_lst = [fun(inducing_inputs, data_points[:, i][:, None]) for fun in derivative_matrix_list]
        d_K_mm__d_sigma_n = self.covariance_obj.get_noise_derivative(points_num=m)
        d_k_i__d_sigma_n = np.zeros((m, 1))
        # # print(k_i.shape)
        d_K_mm__d_theta_lst.append(d_K_mm__d_sigma_n)
        d_k_i__d_theta_lst.append(d_k_i__d_sigma_n)

        # # Variational Lower Bound, estimated by one data point
        loss = 0
        loss = -np.log(sigma_n)/2 - (y_i - k_i.T.dot(K_mm_inv.dot(mu)))**2 / (2 * sigma_n)
        loss += - tilde_K_ii / (2 * sigma_n)
        loss += -np.trace(sigma.dot(Lambda_i))/2
        loss += - np.sum(np.log(np.diag(L))) / N
        loss += - np.trace(sigma.dot(K_mm_inv)) / (2*N)
        loss += - mu.T.dot(K_mm_inv.dot(mu)) / (2*N)

        # Gradient
        grad = np.zeros((len(theta,)))
        for param in range(len(theta)):
            if param != len(theta) - 1:
                cov_derivative = derivative_matrix_list[param]
            else:
                cov_derivative = lambda x, y: self.covariance_obj.get_noise_derivative(points_num=1)
            d_K_mm__d_theta = d_K_mm__d_theta_lst[param]
            d_k_i__d_theta = d_k_i__d_theta_lst[param]
            grad[param] += (y_i - k_i.T.dot(K_mm_inv.dot(mu))) * \
                             (d_k_i__d_theta.T.dot(K_mm_inv) -
                              k_i.T.dot(K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv)))).dot(mu) / sigma_n
            grad[param] += (- cov_derivative(x_i, x_i) +
                              d_k_i__d_theta.T.dot(K_mm_inv.dot(k_i)) -
                              k_i.T.dot(K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv.dot(k_i))))+
                              k_i.T.dot(K_mm_inv.dot(d_k_i__d_theta))
                                ) / (2 * sigma_n)
            grad[param] += np.trace(sigma.dot(
                K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv.dot(k_i.dot(k_i.T.dot(K_mm_inv))))) -
                K_mm_inv.dot(d_k_i__d_theta.dot(k_i.T.dot(K_mm_inv)))
            )) / sigma_n
            grad[param] += - np.trace(K_mm_inv.dot(d_K_mm__d_theta))/ (2*N)
            grad[param] += np.trace(sigma.dot(K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv)))) / (2*N)
            grad[param] += mu.T.dot(K_mm_inv.dot(d_K_mm__d_theta.dot(K_mm_inv.dot(mu)))) / (2*N)

        grad[-1] += tilde_K_ii / (2 * sigma_n**2) - 1 / (2 * sigma_n) + \
                    (y_i - k_i.T.dot(K_mm_inv.dot(mu)))**2 / (2 * sigma_n**2) + \
                    np.trace(sigma.dot(Lambda_i)) / (2 * sigma_n)

        dL_dbeta1 = - y_i / sigma_n * (K_mm_inv.dot(k_i)) + eta_1 / N
        dL_dbeta2 = (-Lambda_i - K_mm_inv / N) / 2 - eta_2 / N

        grad = grad[:, None]

        grad = np.vstack((grad, -dL_dbeta1.reshape(-1)[:, None]))
        grad = np.vstack((grad, dL_dbeta2.reshape(-1)[:, None]))

        self.covariance_obj.set_params(old_params)

        return loss, grad[:, 0]


