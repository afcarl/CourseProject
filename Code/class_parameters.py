from covariance_functions import SquaredExponential, GammaExponential
import numpy as np

class CommonParameters:
    """class, containg the common parameters of the data"""
    def __init__(self, num_of_examples, test_set_size, dim):
        self.n, self.d, self.t = num_of_examples, dim, test_set_size


class ModelParameters:
    """class, containing the hyper-parameters of the model prior distribution"""
    def __init__(self, family, data_seed=None):
        if data_seed == None:
            data_seed = np.random.rand()
        self.cov_obj, self.data_seed = \
            family, data_seed

oracle1 = SquaredExponential(sigma_f=1.0, l=1.0, noise=0.2)
oracle2 = SquaredExponential(sigma_f=1.0, l=1.0, noise=0.2)
# oracle3 = GammaExponential(sigma_f=2.5, l=0.3, gamma=2.1, noise=0.3)
common_params = CommonParameters(num_of_examples=300, dim=2, test_set_size=100)
data_params = ModelParameters(oracle1, data_seed=12)
model_params = ModelParameters(oracle2)
