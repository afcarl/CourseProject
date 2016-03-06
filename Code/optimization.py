import numpy as np


def project_into_bounds(point, bounds):
    """
    Project the given point into the given bounds
    :param bounds:
    :param point:
    :return:
    """
    if bounds is None:
        return point
    low_bounds = [bound[0] for bound in bounds]
    high_bounds = [bound[1] for bound in bounds]
    proj = np.copy(point)
    i = 0
    for coord, l_bound, h_bound in list(zip(point, low_bounds, high_bounds)):
        if not(l_bound is None):
            if coord < l_bound:
                proj[i] = l_bound
        if not(h_bound is None):
            if coord > h_bound:
                proj[i] = h_bound
        i += 1
    return proj


def _linesearch_armiho(fun, point, gradient, bounds=None, step_0=1.0, theta=0.5, eps=1e-2,
                       direction=None, point_loss=None, maxstep=np.inf):
    """
    Line search using Armiho rule
    :param fun: function, being optimized or a tuple (function, gradient)
    :param point: the point of evaluation
    :param direction: direction of the optimization
    :param gradient: gradient at the point point
    :param step_0: initial step length
    :param theta: theta parameter for updating the step length
    :param eps: parameter of the armiho rule
    :param point_loss: fun value at the point point
    :return: (new point, step) — a tuple, containing the chosen step length and
    the next point for the optimization method
    """
    if point_loss is None:
        current_loss = fun(point)
    else:
        current_loss = point_loss
    if direction is None:
        direction = -gradient

    step = step_0/theta
    while step > maxstep:
        step *= theta
    new_point = point + step * direction
    new_point = project_into_bounds(new_point, bounds)
    while fun(new_point) > current_loss + eps * step * direction.dot(gradient):
        step *= theta
        new_point = point + step * direction
        new_point = project_into_bounds(new_point, bounds)
    return new_point, step


def gradient_descent(oracle, point, bounds=None, options=None):
    """
    Gradient descent optimization method
    :param oracle: oracle function, returning the function value and it's gradient, given point
    :param point: point
    :param bounds: bounds on the variables
    :param options: a dictionary, containing the following fields
        'maxiter': maximum number of iterations
        'verbose': a boolean, showing weather or not to print the convergence info
        'print_freq': the frequency of the convergence messages
        'g_tol': the tolerance wrt gradient. If the gradient at the current point is
        smaller than the tollerance, the method stops
        'step_tol' the tolerance wrt the step length. If the step length at current
        iteration is less than tollerance, the method stops.
        'maxstep' is the maximum allowed step length
    default options: {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'g_tol': 1e-5, 'step_tol': 1e-16,
                       'maxstep': 1.0}
    :return: the point with the minimal function value found
    """
    defaul_options = {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'g_tol': 1e-5, 'step_tol': 1e-16,
                       'maxstep': 1.0}
    defaul_options.update(options)
    if 'print_freq' in options.keys():
        defaul_options['verbose'] = True
    options = defaul_options

    step = 1.0
    x = point
    loss_fun = lambda w: oracle(w)[0]

    for i in range(options['maxiter']):
        x = project_into_bounds(x, bounds)
        loss, grad = oracle(x)
        if np.linalg.norm(grad) < options['g_tol']:
            if options['verbose']:
                print("Gradient norm reached the stopping criterion")
            break
        x, step = _linesearch_armiho(fun=loss_fun, gradient=grad, point_loss=loss, bounds=bounds, point=x,
                                     step_0=step, maxstep=options['maxstep'])
        if step < options['step_tol']:
            if options['verbose']:
                print("Step length reached the stopping criterion")
            break
        if not (i % options['print_freq']) and options['verbose']:
            print("Iteration ", i, ":")
            print("\tGradient norm", np.linalg.norm(grad))
            print("\tFunction value", loss)
    return x


def stochastic_gradient_descent(oracle, point, n, bounds=None, options=None):
    """
    Stochastic gradient descent optimization method for finite sums
    :param oracle: an oracle gunction, returning the gradient approximation by one data point,
    given it's index and the point
    :param point:
    :param n: number of training examples
    :param bounds: bounds on the variables
    :param options: a dictionary, containing the following fields
        'maxiter': maximum number of iterations
        'verbose': a boolean, showing weather or not to print the convergence info
        'print_freq': the frequency of the convergence messages
        'batch_size': the size of the mini-batch, used for gradient estimation
        'step0': initial step of the method
        'gamma': a parameter of the step length rule. It should be in (0.5, 1). The smaller it
        is, the more aggressive the method is
        'update_rate': the rate of shuffling the data points
    defaul options: {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'batch_size': 1,
                      'step0': 0.1, 'gamma': 0.55, 'update_rate':1}
    :return:
    """
    defaul_options = {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'batch_size': 1,
                      'step0': 0.1, 'gamma': 0.55, 'update_rate':1}
    defaul_options.update(options)
    if 'print_freq' in options.keys():
        defaul_options['verbose'] = True
    options = defaul_options

    batch_size = options['batch_size']
    step0 = options['step0']
    gamma = options['gamma']

    batch_num = int(n / batch_size)
    if n % batch_size:
        batch_num += 1
    update_rate = options['update_rate']

    indices = np.random.random_integers(0, n-1, (update_rate * n,))
    step = step0
    x = point
    x = project_into_bounds(x, bounds)
    grad = 0
    for epoch in range(options['maxiter']):
        for i in range(n):
            index = indices[i]
            if not (i % batch_size):
                x -= grad * step
                x = project_into_bounds(x, bounds)
                # print('x', x[2])
                grad = oracle(x, index)
            else:
                grad += oracle(x, index)

        if not (epoch % update_rate):
            indices = np.random.random_integers(0, n-1, (update_rate * n,))

        if not (epoch % options['print_freq']) and options['verbose']:
            print("Epoch ", epoch, ":")
            print("\tStep:", step)
        step = step0 / np.power((epoch+1), gamma)
    return x


def check_gradient(oracle, point, print_diff=False):
    """
    Prints the gradient, calculated with the provided function
    and approximated via a finite difference.
    :param oracle: a function, returning the loss and it's grad given point
    :param point: point of calculation
    :param print_diff: a boolean. If true, the method prints all the entries of the true and approx.
    gradients
    :return:
    """
    fun, grad = oracle(point)
    app_grad = np.zeros(grad.shape)
    if print_diff:
        print('Approx.\t\t\t\t Calculated')
    for i in range(point.size):

        point_eps = np.copy(point)
        point_eps[i] += 1e-6
        app_grad[i] = (oracle(point_eps)[0] - fun) * 1e6
        if print_diff:
            print(app_grad[i], '\t', grad[i])
    print('\nDifference between calculated and approximated gradients')
    print(np.linalg.norm(app_grad.reshape(-1) - grad.reshape(-1)))