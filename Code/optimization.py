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
    default options : {'maxiter': 1000, 'print_freq':10, 'verbose': False, 'g_tol': 1e-5, 'step_tol': 1e-16,
                       'maxstep': 1.0}
    :return: the point with the minimal function value found
    """
    if 'maxiter' in options.keys():
        maxiter = options['maxiter']
    else:
        maxiter = 1000
    if 'print_freq' in options.keys():
        print_freq = options['print_freq']
    else:
        print_freq = 10
    if 'verbose' in options.keys():
        verbose = options['verbose']
    elif 'print_freq' in options.keys():
        verbose = True
    else:
        verbose = False
    if 'g_tol' in options.keys():
        g_tol = options['g_tol']
    else:
        g_tol = 1e-5
    if 'step_tol' in options.keys():
        step_tol = options['step_tol']
    else:
        step_tol = 1e-16
    if 'maxstep' in options.keys():
        maxstep = options['maxstep']
    else:
        maxstep = 1.0

    step0 = 0.1
    gamma = 0.55

    step = step0
    x = point
    loss_fun = lambda w: oracle(w)[0]

    for i in range(maxiter):
        x = project_into_bounds(x, bounds)
        print('x', x[1])
        loss, grad = oracle(x)
        if np.linalg.norm(grad) < g_tol:
            if verbose:
                print("Gradient norm reached the stopping criterion")
            break
        # x, step = _linesearch_armiho(fun=loss_fun, gradient=grad, point_loss=loss, bounds=bounds, point=x,
        #                              step_0=step, maxstep=maxstep)
        x -= grad * step
        # x = project_into_bounds(x, bounds)
        if step < step_tol:
            if verbose:
                print("Step length reached the stopping criterion")
            break
        if not (i % print_freq) and verbose:
            print("Iteration ", i, ":")
            print("\tGradient norm", np.linalg.norm(grad))
            print("\tFunction value", loss)
        step = step0 / np.power((i+1), gamma)
    return x
