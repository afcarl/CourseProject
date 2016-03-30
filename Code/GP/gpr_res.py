

class GPRRes:
    """A class, for the objects, returned by gaussian process regression fitting methods"""

    def __init__(self, param_lst, iteration_lst=None, time_lst=None):
        self.params = param_lst
        self.iters = iteration_lst
        if iteration_lst is None:
            self.iters = range(len(self.params))
        self.times = time_lst
        # self.method = gpr_obj.method
        # self.optimizer = gpr_obj.optimizer
        # self.parametrization = gpr_obj.parametrization
        # self.gpr = gpr_obj

    def __str__(self):
        ans = '\nParameter values list:\n'
        ans += str(self.params)
        ans += '\nIteration numbers:\n'
        ans += str(list(self.iters))
        ans += '\nTimes at these iterations:\n'
        ans += str(self.times)
        ans += '\n'
        return ans

    def plot_performance(self, metrics, it_time='i', freq=1):
        print(metrics(self.params[0]))
        y_lst = [metrics(self.params[i]) for i in range(len(self.params)) if not (i % freq)]
        if it_time == 'i':
            x_lst = [self.iters[i] for i in range(len(self.iters)) if not(i%freq)]
        elif it_time == 't':
            x_lst = [self.times[i] for i in range(len(self.times)) if not(i%freq)]
        else:
            raise ValueError('Wrong it_time')
        return x_lst, y_lst
