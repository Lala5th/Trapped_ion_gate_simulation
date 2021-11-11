#!/usr/bin/python3
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

def plot_QuTiP_time(fname):
    global ax, time_slider,axtime

    d = np.load(fname,allow_pickle=True)

    n_num,state_start,nu0,Omega0 = d['metadata']
    n_num = int(n_num)
    state_start = int(state_start)

    os = d['os']
    ts = d['ts']
    s3d = d['s3d']
    max_detuning = os[-1]
    min_detuning = os[0]

    state_data = np.array([[np.array(s,dtype=complex) for s in e] for e in s3d], dtype= complex)
    state_data = np.reshape(np.einsum('ijkl->ikj',np.asarray(state_data,dtype = np.complex128)),(len(os),2,n_num,-1))
    state_data = np.abs(np.einsum('ijkl,ijkl->ijkl',state_data,np.conj(state_data)))

    fig, ax = plt.subplots()
    ps = []
    for i in range(n_num):
        p, = ax.plot(ts,state_data[0,1,i,:],label = f"|e,{i}>")
        p1,= ax.plot(ts,state_data[0,0,i,:],label = f"|g,{i}>",linestyle='--', color = p.get_color())
        ps.append(p1)
        ps.append(p)

    ps = np.reshape(np.array(ps),(-1,2)).T

    ax.legend()
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])q
    # ax.get_xaxis().set_major_formatter(rabi_detuning_format)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("p[1]")
    # ax.set_yscale("logit")
    ax.grid()

    plt.subplots_adjust(bottom=0.25)

    axdetuning = plt.axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(
        ax=axdetuning,
        label="Detuning [$\\Omega$]",
        valmin=min_detuning,
        valmax=max_detuning,
        valinit=min_detuning,
        orientation="horizontal",
        valfmt="%2.4lf"
    )

    for i in range(n_num):
        axdetuning.axvline((i - state_start)*nu0/Omega0,c = ps[0,i].get_color(),linestyle='dashdot')

    def update_time(val):
        id = np.argmin(np.abs(val - os))
        for i in range(2):
            for j in range(n_num):
                ps[i,j].set_ydata(np.abs(state_data[id,i,j,:]))
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)

    plt.show()

def plot_QuTiP_detuning(fname):
    global ax, time_slider,axtime

    d = np.load(fname,allow_pickle=True)

    n_num,state_start,nu0,Omega0 = d['metadata']
    n_num = int(n_num)
    state_start = int(state_start)

    os = d['os']
    ts = d['ts']
    s3d = d['s3d']
    max_time = ts[-1]

    state_data = np.array([[np.array(s,dtype=complex) for s in e] for e in s3d], dtype= complex)
    state_data = np.reshape(np.einsum('ijkl->ikj',np.asarray(state_data,dtype = np.complex128)),(len(os),2,n_num,-1))
    state_data = np.abs(np.einsum('ijkl,ijkl->ijkl',state_data,np.conj(state_data)))

    fig, ax = plt.subplots()
    ps = []
    for i in range(n_num):
        p, = ax.plot(os,state_data[:,1,i,0],label = f"|e,{i}>")
        p1,= ax.plot(os,state_data[:,0,i,0],label = f"|g,{i}>",linestyle='--', color = p.get_color())
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
                ps[i,j].set_ydata(state_data[:,i,j,id])
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)

    plt.show()

def plot_Ground_up_time(fname):
    global ax, time_slider,axtime

    d = np.load(fname)

    n_num,state_start,nu0,Omega0 = d['metadata']
    n_num = int(n_num)
    state_start = int(state_start)

    os = d['os']
    ts = d['ts']
    s3d = d['s3d']
    max_detuning = os[-1]
    min_detuning = os[0]
    s3d = np.einsum('ijkl,ijkl->ijkl',s3d,np.conj(s3d))

    fig, ax = plt.subplots()
    ps = []
    for i in range(n_num):

        p, = ax.plot(ts,np.abs(s3d[0,1,i,:]),label = f"|e,{i}>")
        p1,= ax.plot(ts,np.abs(s3d[0,0,i,:]),label = f"|g,{i}>",linestyle='--', color = p.get_color())
        ps.append(p1)
        ps.append(p)

    ps = np.reshape(np.array(ps),(-1,2)).T

    ax.legend()

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("p[1]")
    # ax.set_yscale("logit")
    ax.grid()

    plt.subplots_adjust(bottom=0.25)

    axdetuning = plt.axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(
        ax=axdetuning,
        label="Detuning [$\\Omega$]",
        valmin=min_detuning,
        valmax=max_detuning,
        valinit=min_detuning,
        orientation="horizontal",
        valfmt="%2.4lf"
    )

    for i in range(n_num):
        axdetuning.axvline((i - state_start)*nu0/Omega0,c = ps[0,i].get_color(),linestyle='dashdot')

    def update_time(val):
        id = np.argmin(np.abs(val - os))
        for i in range(2):
            for j in range(n_num):
                ps[i,j].set_ydata(np.abs(s3d[id,i,j,:]))
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)

    plt.show()

def plot_Ground_up_detuning(fname):
    global ax, time_slider,axtime

    d = np.load(fname)

    n_num,state_start,nu0,Omega0 = d['metadata']
    n_num = int(n_num)
    state_start = int(state_start)

    os = d['os']
    ts = d['ts']
    s3d = d['s3d']
    max_time = ts[-1]

    s3d = np.einsum('ijkl,ijkl->ijkl',s3d,np.conj(s3d))

    fig, ax = plt.subplots()
    ps = []
    for i in range(n_num):

        p, = ax.plot(os,np.abs(s3d[:,1,i,0]),label = f"|e,{i}>")
        p1,= ax.plot(os,np.abs(s3d[:,0,i,0]),label = f"|g,{i}>",linestyle='--', color = p.get_color())
        ps.append(p1)
        ps.append(p)
        ax.axvline((i - state_start)*nu0/Omega0,c = p.get_color(),linestyle='dashdot')

    ps = np.reshape(np.array(ps),(-1,2)).T

    ax.legend()
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


plot_methods = {
    'qutip_time'        : plot_QuTiP_time,
    'qutip_detuning'    : plot_QuTiP_detuning,
    'groundup_time'     : plot_Ground_up_time,
    'groundup_detuning' : plot_Ground_up_detuning
}

if __name__ == '__main__':
    from sys import argv
    import matplotlib as mpl

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

    plt.ion()

    args = argv[1:]
    
    plot_methods[args[0]](args[1])