import sys
import plot_results
args = sys.argv[1:]
get_ipython().run_line_magic('run', '../Aux_scripts/setup_dark_bg.py')
get_ipython().run_line_magic('run', 'plot.py me_seq_dm {args[0]} plot_dm.json')
plot_results.ax.get_figure().set_size_inches([9.58, 4.27])
plot_results.ax.get_figure().tight_layout()
plot_results.ax.get_figure().savefig(args[1],transparent=True)