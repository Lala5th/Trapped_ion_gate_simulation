#!/usr/bin/python3
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

def load_QuTiP(fname):
    
    d = np.load(fname,allow_pickle=True)

    n_num,state_start,nu0,Omega0 = d['metadata']
    n_num = int(n_num)
    state_start = int(state_start)

    os = d['os']
    ts = d['ts']
    s3d = d['s3d']
    max_time = ts[-1]
    max_detuning = os[-1]
    min_detuning = os[0]

    state_data = np.array([[np.array(s,dtype=complex) for s in e] for e in s3d], dtype= complex)
    state_data = np.reshape(np.einsum('ijkl->ikj',np.asarray(state_data,dtype = np.complex128)),(len(os),2,n_num,-1))
    return state_data, os, ts, max_detuning, min_detuning, max_time, nu0, Omega0, n_num, state_start

def load_QuTiP_seq(fname):
    
    d = np.load(fname,allow_pickle=True)
    ts = d['ts']
    s3d = d['s3d']
    n_num, = d['metadata']
    t0s = d['t0s']

    state_data = np.array([np.array(s,dtype=complex) for s in s3d], dtype= complex)
    state_data = np.reshape(np.einsum('ijk->ji',np.asarray(state_data,dtype = np.complex128)),(2,n_num,-1))
    return state_data, ts, n_num, t0s

def load_Ground_up(fname):

    d = np.load(fname)

    n_num,state_start,nu0,Omega0 = d['metadata']
    n_num = int(n_num)
    state_start = int(state_start)

    os = d['os']
    ts = d['ts']
    s3d = d['s3d']
    max_detuning = os[-1]
    max_time = ts[-1]
    min_detuning = os[0]
    return np.array(s3d,dtype=np.complex128), os, ts, max_detuning, min_detuning, max_time, nu0, Omega0, n_num, state_start

def plot_time_scan(data_pack):
    global ax, time_slider,axtime

    state_data, os, ts, max_detuning, min_detuning, _, nu0, Omega0, n_num, state_start = data_pack

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

def plot_detuning_scan(data_pack):
    global ax, time_slider,axtime

    state_data, os, ts, _, _, max_time, nu0, Omega0, n_num, state_start = data_pack

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

def plot_time_scan_projeg(data_pack):
    global ax, time_slider,axtime

    state_data, os, ts, max_detuning, min_detuning, _, nu0, Omega0, n_num, state_start = data_pack

    state_data = np.abs(np.einsum('ijkl,ijkl->ijl',state_data,np.conj(state_data)))

    fig, ax = plt.subplots()
    ps = []
    p, = ax.plot(ts,state_data[0,1,:],label = f"|e>")
    p1,= ax.plot(ts,state_data[0,0,:],label = f"|g>",linestyle='--', color = p.get_color())
    ps.append(p1)
    ps.append(p)

    ps = np.array(ps)

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
        axdetuning.axvline((i - state_start)*nu0/Omega0,linestyle='dashdot',c='r')

    def update_time(val):
        id = np.argmin(np.abs(val - os))
        for i in range(2):
            ps[i].set_ydata(np.abs(state_data[id,i,:]))
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)

    plt.show()

def plot_detuning_scan_projeg(data_pack):
    global ax, time_slider,axtime

    state_data, os, ts, _, _, max_time, nu0, Omega0, n_num, state_start = data_pack

    state_data = np.abs(np.einsum('ijkl,ijkl->ijl',state_data,np.conj(state_data)))

    fig, ax = plt.subplots()
    ps = []
    p, = ax.plot(os,state_data[:,1,0],label = f"|e>")
    p1,= ax.plot(os,state_data[:,0,0],label = f"|g>",linestyle='--', color = p.get_color())
    ps.append(p1)
    ps.append(p)
    for i in range(n_num):
        ax.axvline((i - state_start)*nu0/Omega0,c = 'r',linestyle='dashdot')

    ps = np.array(ps)

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
            ps[i].set_ydata(state_data[:,i,id])
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)

    plt.show()

def plot_seq_scan(data_pack):
    global ax

    state_data, ts, n_num, t0s = data_pack

    state_data = np.abs(np.einsum('ijk,ijk->ijk',state_data,np.conj(state_data)))

    _, ax = plt.subplots()
    for i in range(n_num):
        p, = ax.plot(ts,state_data[1,i,:],label = f"|e,{i}>")
        ax.plot(ts,state_data[0,i,:],label = f"|g,{i}>",linestyle='--', color = p.get_color())

    t0 = 0
    for t in t0s:
        ax.axvline(t0 + t,linestyle='dashdot')
        t0 += t

    ax.legend()
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])q
    # ax.get_xaxis().set_major_formatter(rabi_detuning_format)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("p[1]")
    # ax.set_yscale("logit")
    ax.grid()

    plt.show()

def plot_seq_scan_projeg(data_pack):
    global ax

    state_data, ts, _, t0s = data_pack

    state_data = np.abs(np.einsum('ijk,ijk->ik',state_data,np.conj(state_data)))

    _, ax = plt.subplots()
    p, = ax.plot(ts,state_data[1,:],label = f"|e>")
    ax.plot(ts,state_data[0,:],label = f"|g>",linestyle='--', color = p.get_color())

    t0 = 0
    for t in t0s:
        ax.axvline(t0 + t,linestyle='dashdot')
        t0 += t

    ax.legend()
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])q
    # ax.get_xaxis().set_major_formatter(rabi_detuning_format)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("p[1]")
    # ax.set_yscale("logit")
    ax.grid()

    plt.show()

def plot_seq_scan_Fockexp(data_pack):
    global ax

    state_data, ts, n_num, t0s = data_pack

    state_data = np.abs(np.einsum('ijk,ijk->ijk',state_data,np.conj(state_data)))

    exp = np.diag([i for i in range(n_num)])
    state_data = np.einsum('ijk,jl->ilk',state_data,exp)
    state_data = np.einsum('ijk->k',state_data)

    _, ax = plt.subplots()
    ax.plot(ts,state_data,label = "<$a^\\dagger{}a$>",linestyle='--')

    t0 = 0
    for t in t0s:
        ax.axvline(t0 + t,linestyle='dashdot')
        t0 += t

    ax.legend()
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])q
    # ax.get_xaxis().set_major_formatter(rabi_detuning_format)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("E[1]")
    # ax.set_yscale("logit")
    ax.grid()

    plt.show()

plot_methods = {
    'qutip_time'            : [load_QuTiP,plot_time_scan],
    'qutip_detuning'        : [load_QuTiP,plot_detuning_scan],
    'qutip_time_projeg'     : [load_QuTiP,plot_time_scan_projeg],
    'qutip_detuning_projeg' : [load_QuTiP,plot_detuning_scan_projeg],
    'qutip_seq'             : [load_QuTiP_seq,plot_seq_scan],
    'qutip_seq_projeg'      : [load_QuTiP_seq,plot_seq_scan_projeg],
    'qutip_seq_fockexp'     : [load_QuTiP_seq,plot_seq_scan_Fockexp],
    'groundup_time'         : [load_Ground_up,plot_time_scan],
    'groundup_detuning'     : [load_Ground_up,plot_detuning_scan]
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
    
    last = args[1]
    for m in plot_methods[args[0]]:
        last = m(last)