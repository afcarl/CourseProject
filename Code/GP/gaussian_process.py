import numpy as np
from abc import ABCMeta, abstractmethod
from GP.optimization import _eig_val_correction
import numbers

class GP:
    """
    An abstract class, a base class for GPR and GPC
    """
    __metaclass__ = ABCMeta

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
        res = np.random.multivariate_normal(mean_vector, cov_mat)
        return res

    @staticmethod
    def sample_for_matrices(mean_vec, cov_mat, rnd=None):
        """
        :param mean_vec: mean vector
        :param cov_mat: cavariance matrix
        :return: sample gaussian process values at given points
        """
        if not (isinstance(mean_vec, np.ndarray) and
                isinstance(cov_mat, np.ndarray)):
            raise TypeError("points must be a numpy array")
        if rnd is None or (isinstance(rnd, bool) and rnd == False):
            upper_bound = mean_vec + 3 * np.sqrt(np.diagonal(cov_mat).reshape(mean_vec.shape))
            lower_bound = mean_vec - 3 * np.sqrt(np.diagonal(cov_mat).reshape(mean_vec.shape))
            return mean_vec, upper_bound, lower_bound
        else:
            if isinstance(rnd, numbers.Number):
                np.random.seed(rnd)
            y = np.random.multivariate_normal(mean_vec.reshape(-1), cov_mat)
            return y

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
        mat = np.zeros((int(k), int(k)))
        mat[indices] = vec.reshape(-1)
        return mat

    @staticmethod
    def _get_inv_logdet_cholesky(mat):
        try:
            L = np.linalg.cholesky(mat)
        except:
            L = np.linalg.cholesky(_eig_val_correction(mat, eps=1e-1))
        L_inv = np.linalg.inv(L)
        mat_inv = L_inv.T.dot(L_inv)
        mat_logdet = np.sum(np.log(np.diag(L))) * 2
        return mat_inv, mat_logdet