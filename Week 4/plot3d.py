import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from sys import argv
from mpl_toolkits.mplot3d import Axes3D

plt.ion()

args = argv[1:]

if __name__ == '__main__':

    d = np.load(args[0])
    if(len(args) > 1):
        n_num = int(args[1])
        state_start = int(args[2])

        Omega0 = float(args[3])
        nu0 = float(args[4])
    else:
        n_num,state_start,nu0,Omega0 = d['metadata']
        n_num = int(n_num)
        state_start = int(state_start)

    os = np.array([d['os']]*len(d['ts'])).T
    ts = np.array([d['ts']]*len(d['os']))
    s3d = d['s3d']
    max_time = ts[-1,-1]
    max_detuning = os[-1,-1]
    min_detuning = os[0,0]

    rabi_detuning_format = lambda x, pos = None : f"{x} $\\Omega$" if x!=0 else "0"

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(n_num):
        # col = [0.3,1,0.3]
        # if (i < state_start):
        #     col = [1 - (state_start - i - 1)/(state_start),0.2,0.2]
        # elif (i > state_start):
        #     col = [0.2,0.2,(i - state_start + 1)/(n_num - state_start)]
        ax.plot_wireframe(os,ts,np.abs(s3d[:,1,i,:]),label = f"|e,{i}>")
        ax.plot_wireframe(os,ts,np.abs(s3d[:,0,i,:]),label = f"|g,{i}>")
        # ax.axvline((i - state_start)*nu0/Omega0,c = p.get_color(),linestyle='dashdot')

    ax.legend()
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])q
    # ax.get_xaxis().set_major_formatter(rabi_detuning_format)
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("p[1]")
    # ax.set_yscale("logit")
    ax.grid()
    plt.show()
