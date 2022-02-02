from plot_results import plot_methods
import plot_results
import matplotlib.pyplot as plt
from sys import argv
args = argv[1:]
plot_results.args = args

if __name__ == '__main__':
    import matplotlib as mpl

    rcparams = {
    # 'axes.titlesize'    : 18,
    # 'axes.labelsize'    : 16,
    # 'xtick.labelsize'   : 12,
    # 'ytick.labelsize'   : 12,
    # 'legend.fontsize'   : 12,
    'font.size'         : 16
    }
    for e in rcparams.keys():
        mpl.rcParams[e] = rcparams[e]

    plt.ion()

    
    last = args[1]
    for m in plot_methods[args[0]]:
        last = m(last)