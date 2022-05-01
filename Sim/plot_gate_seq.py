import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import plot_results
from plot_results import plot_methods
import plot_results
import matplotlib.pyplot as plt
from sys import argv
args = argv[1:]

for dn,fn in [("ms_gate_seq_or","MS_Gate_seq_OR"),("cardioid_seq","Cardioid_seq"),("cardioid_seq_or","Cardioid_seq_OR"),("SC2_seq","SC2_seq"),("SC2_seq_or","SC2_seq_OR"),("Custom_SC2_seq","Custom_SC2_seq"),("Custom_SC2_seq_or","Custom_SC2_seq_OR")]:
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

    
    plot_results.args = ["me_seq_dm", f"{dn}.npz", "plot_dm.json"]
    args = plot_results.args
    last = args[1]
    for m in plot_methods[args[0]]:
        last = m(last)
    
    plot_results.ax.get_figure().set_size_inches([9.58, 4.27])
    plot_results.ax.get_figure().tight_layout()
    plot_results.ax.get_figure().savefig(f"../Viva/{fn}.pdf",transparent=True)

