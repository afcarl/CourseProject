import numpy as np
import matplotlib as mpl

# def figsize(scale):
#     fig_width_pt = 460.72124                          # Get this from LaTeX using \the\textwidth
#     inches_per_pt = 1.0/72.27                       # Convert pt to inch
#     golden_mean = (np.sqrt(5.0)-1.0)/2.0 + 0.2         # Aesthetic ratio (you could change this)
#     fig_width = (fig_width_pt*inches_per_pt + 2.5)*scale    # width in inches
#     fig_height = fig_width * golden_mean           # height in inches
#     fig_size = [fig_width, fig_height]
#     return fig_size
#
# pgf_with_latex = {                      # setup matplotlib to use latex for output
#     "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
#     "text.usetex": True,                # use LaTeX to write all text
#     "font.family": "serif",
#     "font.serif": [],                  # blank entries should cause plots to inherit fonts from the document
#     "font.sans-serif": [],
#     "font.monospace": [],
#     "axes.labelsize": 10,               # LaTeX default is 10pt font.
#     "text.fontsize": 10,
#     "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
#     "xtick.labelsize": 8,
#     "ytick.labelsize": 8,
#     "figure.figsize": figsize(0.35),     # default fig size of 0.9 textwidth
#     "pgf.preamble": [
#         r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts because your computer can handle it :)
#         r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
#         ]
#     }
# mpl.rcParams.update(pgf_with_latex)

from matplotlib import pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import r2_score
from GP.gaussian_process_regression import GPR
from GP.plotting import plot_reg_data, plot_predictive
from GP.covariance_functions import SquaredExponential, Matern, GammaExponential
from matplotlib.mlab import griddata
from matplotlib2tikz import save

data_params = np.array([1.0, 0.15, 0.1])
data_covariance_obj = SquaredExponential(data_params)

# model_params = np.array([1.0, 1., 0.1])
# model_covariance_obj = SquaredExponential(model_params)
# model_params = np.array([1.0, 0.1, 0.5, 0.1])
# model_covariance_obj = GammaExponential(model_params)
model_params = np.array([0.3, 0.2, .1])
model_covariance_obj = SquaredExponential(model_params)

gp = GPR(data_covariance_obj)
num = 50
test_num = 100
dim = 1
seed = 10
ind_inputs_num = 5
max_iter = 200
batch_size = 50

method = 'svi'  # possible methods: 'brute', 'vi', 'means', 'svi'
parametrization = 'natural'  # possible parametrizations for svi method: cholesky, natural
optimizer = 'SG'
# possible optimizers: 'AdaDelta', 'FG', 'L-BFGS-B' for cholesky-svi;
# 'L-BFGS' and 'Projected Newton' for 'means' and 'vi'



# sag_options = {'maxiter':max_iter, 'batch_size': 20, 'print_freq': 10}
sag_options = {'mydisp': False, 'print_freq': 1, 'step_rate': 0.5,
                'maxiter': 5, 'batch_size':20}
fg_options = {'maxiter':max_iter, 'print_freq': 100}
lbfgsb_options = {'maxiter': max_iter, 'disp': False, 'mydisp': True}
sg_options = {'maxiter':max_iter, 'batch_size': batch_size, 'print_freq': 100, 'step0': 1e-4, 'gamma': 0.55}
# sg_options = {'mydisp': False, 'print_freq': 1, 'step_rate': 1e-2,
#                 'maxiter': 50, 'batch_size':20}
projected_newton_options = {'maxiter':max_iter, 'print_freq': 1}

# Generating data points
np.random.seed(seed)
x_tr = np.random.rand(dim, num)
if dim == 1:
    x_test = np.linspace(0, 1, test_num)
    x_test = x_test.reshape(1, test_num)
else:
    x_test = np.random.rand(dim, test_num)

y_tr, y_test = gp.generate_data(x_tr, x_test, seed=seed)

if method == 'brute':
    new_gp = GPR(model_covariance_obj)
    # res = new_gp.fit(x_tr, y_tr, max_iter=max_iter)
    predicted_y_test, high, low = new_gp.predict(x_test, x_tr, y_tr)

elif method == 'means' or method == 'vi':
    model_covariance_obj = SquaredExponential(model_params)
    new_gp = GPR(model_covariance_obj, method=method, optimizer=optimizer)
    res = new_gp.fit(x_tr, y_tr, num_inputs=ind_inputs_num,  optimizer_options=lbfgsb_options)
    # res = new_gp.fit(x_tr, y_tr, num_inputs=ind_inputs_num,  optimizer_options=projected_newton_options)
    inducing_points, mean, cov = new_gp.inducing_inputs
    predicted_y_test, high, low = new_gp.predict(x_test)

elif method == 'svi':
    model_covariance_obj = SquaredExponential(model_params)
    if parametrization == 'natural':
        opts = sg_options
    else:
        if optimizer == 'L-BFGS-B':
            opts = lbfgsb_options
        elif optimizer == 'AdaDelta':
            opts = sag_options
        else:
            opts = sg_options
    new_gp = GPR(model_covariance_obj, method=method, parametrization=parametrization, optimizer=optimizer)
    res = new_gp.fit(x_tr, y_tr, num_inputs=ind_inputs_num, optimizer_options=opts)
    inducing_points, mean, cov = new_gp.inducing_inputs
    predicted_y_test, high, low = new_gp.predict(x_test)

print(new_gp.covariance_obj.get_params())
print(r2_score(y_test, predicted_y_test))

if dim == 1:
    plot_reg_data(x_tr, y_tr, 'k+', mew=1, ms=8)
    plot_predictive(x_test, predicted_y_test, high, low)
    # plot_reg_data(x_test, y_test, 'g-')
    if method != 'brute':
        plot_reg_data(inducing_points, mean, 'ro', markersize=8)
    # plt.title("Predictive distribution")
    # plt.title("Matern covariance function, $\\nu = 1$")
    # plt.savefig('pictures/')
    plt.show()

elif dim == 2:
    xi = np.linspace(0, 1, 100)
    yi = np.linspace(0, 1, 100)
    zi = griddata(x_tr[0, :], x_tr[1, :], y_tr[:, 0], xi, yi, interp='linear')
    # plt.plot(x_tr[0], x_tr[1], 'k+')
    if method != 'brute':
        plot_reg_data(inducing_points[0, :], inducing_points[1, :], 'ro', markersize=5)

    plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k')
    plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow,
                      vmax=abs(zi).max(), vmin=-abs(zi).max())
    plt.title('A two-dimensional gaussian process')
    plt.colorbar()
    plt.axis('off')
    # plt.savefig('pictures/')
    plt.show()
