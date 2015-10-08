import unittest
import numpy as np
import numpy.testing as npt

from gaussian_process import GaussianProcess
from covariance_functions import SquaredExponential


class TestMyFunctions(unittest.TestCase):
    def setUp(self):
        self.mean = lambda x: 0
        self.cov = SquaredExponential(sigma_f=2.0, l=1.0, noise=1.2)
        self.gp = GaussianProcess(self.cov, self.mean)

    def test_init_wrong_inputs(self):
        self.assertRaises(TypeError, GaussianProcess, self.mean, self.mean)
        self.assertRaises(TypeError, GaussianProcess, self.cov, 0)

    def test_generate_data_wrong_inputs(self):
        self.assertRaises(ValueError, self.gp.generate_data, 1, 1, 'sass')
        self.assertRaises(ValueError, self.gp.generate_data, 1, 0, 'class')
        self.assertRaises(ValueError, self.gp.generate_data, 0, 1, 'class')

    def test_sample_wrong_inputs(self):
        self.assertRaises(TypeError, self.gp.sample, self.mean, self.cov, [1, 2, 3])
        self.assertRaises(TypeError, self.gp.sample, 0, self.cov, np.array([0, 1, 2]))
        self.assertRaises(TypeError, self.gp.sample, self.mean, [], np.array([0, 1, 2]))


if __name__ == "__main__":
    wrong_input_suite = unittest.TestSuite()
    wrong_input_suite.addTest(TestMyFunctions.test_init_wrong_inputs)
    wrong_input_suite.addTest(TestMyFunctions.test_generate_data_wrong_inputs)
    wrong_input_suite.addTest(TestMyFunctions.test_sample_wrong_inputs)
    wrong_input_suite.run()

