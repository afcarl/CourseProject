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


def plot_lists(file_name, title, show=False):

    color = '-yo'
    name = 'svi-natural'
    x_lst, y_lst = np.load('saved_lists/svi-natural_x.npy'), np.load('saved_lists/svi-natural_y.npy')
    plt.plot(x_lst, y_lst, color, label=name)

    color = '-kx'
    name = 'vi-means'
    x_lst, y_lst = np.load('saved_lists/vi-means_x.npy'), np.load('saved_lists/vi-means_y.npy')
    plt.plot(x_lst, y_lst, color, label=name)

    color = '-go'
    name = 'svi-LBFGS-B'
    x_lst, y_lst = np.load('saved_lists/svi-lbfgs_x.npy'), np.load('saved_lists/svi-lbfgs_y.npy')
    plt.plot(x_lst, y_lst, color, label=name)

    plt.ylabel('$R^2$-score on test data')

    plt.legend()
    plt.title(title)
    plt.savefig('../plots/vi_vs_svi/'+file_name + '.pgf')
    if show:
        plt.show()

if __name__ == '__main__':
    name = '1e5_sg_lbfgs'
    title = 'Year Prediction, n = 100000, d = 90, m=1000'
    plot_lists(name, title)