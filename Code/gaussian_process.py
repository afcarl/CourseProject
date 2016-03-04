import numpy as np
import time
import scipy.optimize as op
from abc import ABCMeta, abstractmethod


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