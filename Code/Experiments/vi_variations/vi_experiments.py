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


def run_methods(train_points, train_targets, test_points, test_targets,
                model_parameters, optimizer_options, file_name, ind_num, title, show=False):
    method = 'means'

    means = KMeans(n_clusters=ind_num)
    means.fit(train_points.T)
    inputs = means.cluster_centers_.T

    for optimizer, color, opts in zip(['L-BFGS-B', 'Projected Newton'], ['-kx', '-mx'],
                                      optimizer_options):
        print('Optimizer', optimizer)
        model_covariance_obj = SquaredExponential(np.copy(model_parameters))
        new_gp = GPR(model_covariance_obj, method=method, optimizer=optimizer)
        res = new_gp.fit(train_points, train_targets, num_inputs=ind_num, optimizer_options=opts, inputs=inputs)
        name = optimizer
        metric = lambda w: new_gp.get_prediction_quality(w, train_points, train_targets, test_points, test_targets)
        x_lst, y_lst = res.plot_performance(metric, 'i', freq=1)
        plt.plot(x_lst, y_lst, color, label=name)

    plt.xlabel('Epoch')
    plt.ylabel('$R^2$-score on test data')
    plt.legend()
    plt.title(title)
    plt.savefig('../Plots/vi_variations/'+file_name + '.pgf')
    if show:
        plt.show()