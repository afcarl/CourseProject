import numpy as np
import matplotlib as mpl

def figsize(scale):
    fig_width_pt = 460.72124                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0 + 0.28         # Aesthetic ratio (you could change this)
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
    "legend.fontsize": 10,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.45),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts because your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        ]
    }
mpl.rcParams.update(pgf_with_latex)
# mpl.use("pgf")

from matplotlib import pyplot as plt

# data_name, max_time, lims = 'german', 5, [.65, .8]
# data_name, max_time, lims = 'ijcnn', 200, [.8, .95]
# data_name, max_time, lims = 'magic telescope', 45, [.7, .9]
# data_name, max_time, lims = 'svmguide', 2, [.85, 1.]
data_name, max_time, lims = 'cod-rna', 1200, [.93, .97]
# data_name, max_time, lims = 'skin_nonskin', 200, [.95, 1.]

with open('../plots/vi_vs_svi_class/'+ data_name +'_title.txt') as f:
    title = f.read()
svi_x_lst = np.load('../plots/vi_vs_svi_class/'+ data_name +'_svi_x.npy')
svi_y_lst = np.load('../plots/vi_vs_svi_class/'+ data_name +'_svi_y.npy')
vi_x_lst = np.load('../plots/vi_vs_svi_class/'+ data_name +'_vi_x.npy')
vi_y_lst = np.load('../plots/vi_vs_svi_class/'+ data_name +'_vi_y.npy')
vi_t_x_lst = np.load('../plots/vi_vs_svi_class/'+ data_name +'_vi_t_x.npy')
vi_t_y_lst = np.load('../plots/vi_vs_svi_class/'+ data_name +'_vi_t_y.npy')
svi_ad_x_lst = np.load('../plots/vi_vs_svi_class/'+ data_name +'_ad_x.npy')
svi_ad_y_lst = np.load('../plots/vi_vs_svi_class/'+ data_name +'_ad_y.npy')

svi_lim = np.max(np.where(np.array(svi_x_lst) < max_time)) + 1
vi_lim = np.max(np.where(np.array(vi_x_lst) < max_time)) + 1
vi_t_lim = np.max(np.where(np.array(vi_t_x_lst) < max_time)) + 1
svi_ad_lim = np.max(np.where(np.array(svi_ad_x_lst) < max_time)) + 1

plt.plot(svi_x_lst[1:svi_lim], svi_y_lst[1:svi_lim], '-gx', label='svi-L-BFGS-B')
plt.plot(svi_ad_x_lst[1:svi_ad_lim], svi_ad_y_lst[1:svi_ad_lim], '-yx', label='svi-AdaDelta')
plt.plot(vi_x_lst[1:vi_lim], vi_y_lst[1:vi_lim], '-bx', label='vi-JJ')
plt.plot(vi_t_x_lst[1:vi_t_lim], vi_t_y_lst[1:vi_t_lim], '-mx', label='vi-Taylor')
plt.ylim(lims)
plt.legend(loc=4)
plt.title(title)
plt.xlabel('Time (seconds)')
plt.ylabel('Validation Accuracy')
plt.show()