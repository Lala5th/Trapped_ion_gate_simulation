import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
from sys import argv
import qutip as qtip
import matplotlib as mpl

plt.ion()

rcparams = {
# 'axes.titlesize'    : 18,
# 'axes.labelsize'    : 16,
# 'xtick.labelsize'   : 12,
# 'ytick.labelsize'   : 12,
# 'legend.fontsize'   : 12,
'font.size'         : 14
}
for e in rcparams.keys():
    mpl.rcParams[e] = rcparams[e]

args = argv[1:]

if __name__ == '__main__':

    d = np.load(args[0],allow_pickle=True)
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

    rabi_detuning_format = lambda x, pos = None : f"{x} $\\Omega$" if x!=0 else "0"

    state_data = np.zeros((len(os),2,n_num,len(ts)))

    # proj_e = qtip.basis(2,1)*qtip.basis(2,1).dag()
    # proj_g = qtip.basis(2,0)*qtip.basis(2,0).dag()
    # for i in range(n_num):
    #     proj_M = qtip.basis(n_num,i)*qtip.basis(n_num,i).dag()
    #     proj_e_c = qtip.tensor(proj_e,proj_M)
    #     proj_g_c = qtip.tensor(proj_g,proj_M)
    #     for j,o in enumerate(s3d):
    #         print(j,"/",len(os))
    #         for k,t_s in enumerate(o):
    #             t_s = qtip.Qobj(t_s,[[2,7],[1,1]])
    #             state_data[j,0,i,k] = abs((t_s.dag()*proj_g_c*t_s)[0,0])
    #             state_data[j,1,i,k] = abs((t_s.dag()*proj_e_c*t_s)[0,0])

    state_data = np.reshape(np.einsum('ijkl->ikj',np.asarray(s3d,dtype = np.complex128)),(len(os),2,n_num,-1))
    state_data = np.abs(np.einsum('ijkl,ijkl->ijkl',state_data,np.conj(state_data)))

    fig, ax = plt.subplots()
    ps = []

    p, = ax.plot(os,np.einsum('ij->i',state_data[:,1,:,0]),label = f"|e>")
    p1,= ax.plot(os,np.einsum('ij->i',state_data[:,0,:,0]),label = f"|g>",linestyle='--', color = p.get_color())
    ps.append(p1)
    ps.append(p)
    for i in range(n_num):
        ax.axvline((i - state_start)*nu0/Omega0,c = p.get_color(),linestyle='dashdot')

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
            ps[i].set_ydata(np.einsum('ij->i',state_data[:,i,:,id]))
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)

    plt.show()
