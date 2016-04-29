import numpy as np
import matplotlib as mpl

def figsize(scale):
    fig_width_pt = 460.72124                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0 + 0.2         # Aesthetic ratio (you could change this)
    fig_width = (fig_width_pt*inches_per_pt + 2.5)*scale    # width in inches
    fig_height = fig_width * golden_mean           # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "text.fontsize": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.5),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts because your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)

from matplotlib import pyplot as plt
from GP.covariance_functions import SquaredExponential
from GP.gaussian_process_regression import GPR
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

def run_methods(train_points, train_targets, test_points, test_targets,
                model_parameters, m_list, file_name, title, show=False, full=True, vi=True):

    method = 'means'
    optimizer = 'L-BFGS-B'
    max_iter = 50
    options = {'maxiter': max_iter, 'disp': False, 'mydisp': True}

    means_r2 = []
    vi_r2 = []

    for m in m_list:
        print('m:', m)
        print('Finding means...')
        means = KMeans(n_clusters=m, n_init=1, max_iter=20)
        means.fit(train_points.T)
        inputs = means.cluster_centers_.T
        print('...found')

        model_covariance_obj = SquaredExponential(np.copy(model_parameters))
        new_gp = GPR(model_covariance_obj, method='means', optimizer=optimizer)
        res = new_gp.fit(train_points, train_targets, num_inputs=m, optimizer_options=options, inputs=inputs)
        predicted_y_test, _, _ = new_gp.predict(test_points)
        means_r2.append(r2_score(test_targets, predicted_y_test))

        if vi:
            model_covariance_obj = SquaredExponential(np.copy(model_parameters))
            new_gp = GPR(model_covariance_obj, method='vi', optimizer=optimizer)
            res = new_gp.fit(train_points, train_targets, num_inputs=m, optimizer_options=options, inputs=inputs)
            predicted_y_test, _, _ = new_gp.predict(test_points)
            vi_r2.append(r2_score(test_targets, predicted_y_test))

    if full:
        model_covariance_obj = SquaredExponential(np.copy(model_parameters))
        new_gp = GPR(model_covariance_obj, method='brute')
        res = new_gp.fit(train_points, train_targets, max_iter=max_iter)
        predicted_y_test, _, _ = new_gp.predict(test_points, train_points, train_targets)
        brute_r2 = r2_score(test_targets, predicted_y_test)

    plt.plot(range(len(m_list)), means_r2, '-kx', label='vi-means')
    if vi:
        plt.plot(range(len(m_list)), vi_r2, '-rx', label='vi')
    if full:
        plt.plot(range(len(m_list)), len(m_list) * [brute_r2], '--g', label='full GP')

    plt.xticks(range(len(m_list)), m_list)
    plt.xlabel('m')
    plt.ylabel('$R^2$-score on test data')
    # plt.ylim(0.5, 1)
    plt.legend(loc=4)
    plt.title(title)
    plt.savefig('../Plots/inducing_inputs/'+file_name + '.pgf')
    if show:
        plt.show()