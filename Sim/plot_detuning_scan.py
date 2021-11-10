import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from sys import argv

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

    os = d['os']
    ts = d['ts']
    s3d = d['s3d']
    max_time = ts[-1]
    max_detuning = os[-1]

    s3d = np.einsum('ijkl,ijkl->ijkl',s3d,np.conj(s3d))

    rabi_detuning_format = lambda x, pos = None : f"{x} $\\Omega$" if x!=0 else "0"

    fig, ax = plt.subplots()
    ps = []
    for i in range(n_num):
        # col = [0.3,1,0.3]
        # if (i < state_start):
        #     col = [1 - (state_start - i - 1)/(state_start),0.2,0.2]
        # elif (i > state_start):
        #     col = [0.2,0.2,(i - state_start + 1)/(n_num - state_start)]
        p, = ax.plot(os,np.abs(s3d[:,1,i,0]),label = f"|e,{i}>")
        p1,= ax.plot(os,np.abs(s3d[:,0,i,0]),label = f"|g,{i}>",linestyle='--', color = p.get_color())
        ps.append(p1)
        ps.append(p)
        ax.axvline((i - state_start)*nu0/Omega0,c = p.get_color(),linestyle='dashdot')

    ps = np.reshape(np.array(ps),(-1,2)).T

    ax.legend()
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])
    # ax.get_xaxis().set_major_formatter(rabi_detuning_format)
    ax.set_xlabel("Detuning [$\\Omega$]")
    ax.set_ylabel("p[1]")
    # ax.set_yscale("logit")
    ax.grid()

    plt.subplots_adjust(bottom=0.25)

    axtime = plt.axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(
        ax=axtime,
        label="Time [s]",
        valmin=0,
        valmax=max_time,
        valinit=0,
        orientation="horizontal",
        valfmt="%2.4lf"
    )

    def update_time(val):
        id = np.argmin(np.abs(val - ts))
        for i in range(2):
            for j in range(n_num):
                ps[i,j].set_ydata(np.abs(s3d[:,i,j,id]))
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)

    plt.show()
