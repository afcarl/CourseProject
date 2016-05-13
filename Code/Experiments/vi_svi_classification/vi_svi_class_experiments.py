import numpy as np
# import matplotlib as mpl
#
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
#     "figure.figsize": figsize(0.5),     # default fig size of 0.9 textwidth
#     "pgf.preamble": [
#         r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts because your computer can handle it :)
#         r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
#         ]
#     }
# mpl.rcParams.update(pgf_with_latex)

from matplotlib import pyplot as plt
from GP.covariance_functions import SquaredExponential
from GP.gaussian_process_classification import GPC
# from GP.gaussian_process_regression import GPR
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

def run_methods(train_points, train_targets, test_points, test_targets,
                model_parameters, m_list, file_name, title, show=False):

    maxiter = 200
    max_out_iter = 100
    opts = {'maxiter':maxiter, 'mydisp': True}
    vi_means_errors = []
    svi_means_errors = []

    for m in m_list:
        print('m:', m)
        print('Finding means...')
        means = KMeans(n_clusters=m, n_init=1, max_iter=20, random_state=241)
        means.fit(train_points.T)
        inputs = means.cluster_centers_.T
        print('...found')

        model_covariance_obj = SquaredExponential(np.copy(model_parameters))
        new_gp = GPC(model_covariance_obj, method='svi', hermgauss_deg=100)
        new_gp.fit(train_points, train_targets, inputs=inputs, optimizer_options=opts)
        predicted_y_test = new_gp.predict(test_points)
        svi_means_errors.append(1. - np.sum(test_targets != predicted_y_test) / test_targets.size)

        model_covariance_obj = SquaredExponential(np.copy(model_parameters))
        new_gp = GPC(model_covariance_obj, method='vi')
        new_gp.fit(train_points, train_targets, inputs=inputs, optimizer_options=opts, max_out_iter=max_out_iter)
        predicted_y_test = new_gp.predict(test_points)
        vi_means_errors.append(1. - np.sum(test_targets != predicted_y_test) / test_targets.size)

    plt.plot(range(len(m_list)), svi_means_errors, '-kx', label='svi-classification')
    plt.plot(range(len(m_list)), vi_means_errors, '-mx', label='vi-classification')

    plt.xticks(range(len(m_list)), m_list)
    plt.xlabel('m')
    plt.ylabel('Accuracy on test data')
    # plt.ylim(0.5, 1)
    plt.legend(loc=4)
    plt.title(title)
    # plt.savefig('../Plots/inducing_inputs/'+file_name + '.pgf')
    if show:
        plt.show()