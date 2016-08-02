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
    # "font.family": "serif",
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
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
# mpl.use("pgf")

from matplotlib import pyplot as plt

from GP.covariance_functions import SquaredExponential
from GP.gaussian_process_classification import GPC

from sklearn.cluster import KMeans


def run_methods(train_points, train_targets, test_points, test_targets,
                model_parameters, optimizer_options, max_out_iter, file_name, ind_num, title, show=False, time_it='t',
                adadelta=True, vi=True, svi=True):

    print('Finding means...')
    means = KMeans(n_clusters=ind_num, n_init=3, max_iter=100, random_state=241)
    means.fit(train_points.T)
    inputs = means.cluster_centers_.T
    print('...found')

    if svi:
        #svi-L-BFGS-c method
        method = 'svi'
        color = '-gx'
        name = 'svi-L-BFGS-B-c'
        print(name)
        opts = optimizer_options[0]
        model_covariance_obj = SquaredExponential(np.copy(model_parameters))
        new_gp = GPC(model_covariance_obj, method=method, hermgauss_deg=100)
        res = new_gp.fit(train_points, train_targets, inputs=inputs, optimizer_options=opts)

        metric = lambda w: new_gp.get_prediction_quality(w, test_points, test_targets)
        # np.save('params.npy', np.array(res.params))
        # np.save('means.npy', inputs)
        x_lst, y_lst = res.plot_performance(metric, 't', freq=10)
        plt.plot(x_lst, y_lst, color, label=name)
        np.save(file_name + '_svi_x_lst.npy', x_lst)
        np.save(file_name + '_svi_y_lst.npy', y_lst)
        #
        print(y_lst[-1])
        # np.save('mnist_svi_x_lst.npy', x_lst)
        # np.save('mnist_svi_y_lst.npy', y_lst)

    if vi:
        #vi-means-c method
        name = 'vi-means-c'
        method = 'vi'
        color = '-bx'
        print(name)
        opts = optimizer_options[1]

        model_covariance_obj = SquaredExponential(np.copy(model_parameters))
        new_gp = GPC(model_covariance_obj, method=method)
        res = new_gp.fit(train_points, train_targets, inputs=inputs, max_out_iter=max_out_iter, optimizer_options=opts)
        metric = lambda w: new_gp.get_prediction_quality(w, test_points, test_targets)
        x_lst, y_lst = res.plot_performance(metric, time_it, freq=1)
        plt.plot(x_lst, y_lst, color, label=name)
        np.save(file_name + '_vi_x_lst.npy', x_lst)
        np.save(file_name + '_vi_y_lst.npy', y_lst)

    if adadelta:
        #svi-AdaDelta-c method
        method = 'svi'
        color = '-yx'
        name = 'svi-AdaDelta-c'
        print(name)
        opts = optimizer_options[2]
        model_covariance_obj = SquaredExponential(np.copy(model_parameters))
        new_gp = GPC(model_covariance_obj, method=method, hermgauss_deg=100)
        res = new_gp.fit(train_points, train_targets, inputs=inputs, optimizer_options=opts)

        metric = lambda w: new_gp.get_prediction_quality(w, test_points, test_targets)
        x_lst, y_lst = res.plot_performance(metric, 't', freq=1)
        plt.plot(x_lst, y_lst, color, label=name)


    # print(x_lst[-1])
    if time_it == 't':
        plt.xlabel('Seconds')
    else:
        plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.legend(loc=4)
    plt.title(title)
    # plt.savefig('test.pdf')
    # plt.savefig('test.pgf')
    # plt.savefig('../plots/vi_vs_svi_class/'+file_name + '.pdf')
    if show:
        plt.show()