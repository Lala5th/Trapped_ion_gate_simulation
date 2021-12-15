#!/usr/bin/python3
from matplotlib import lines
import numpy as np
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import json
from scipy.integrate import quad, dblquad
from scipy.linalg import expm
import scipy.constants as const
from functools import lru_cache
from matplotlib import colors, cm
from scipy.optimize import minimize
from numpy.fft import fft, fftfreq, fftshift

def multiple_formatter(denominator=2, number=np.pi, latex='\\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter

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
    n_num, n_ion = d['metadata']
    t0s = d['t0s']

    state_data = np.array([np.array(s,dtype=complex) for s in s3d], dtype= complex)
    state_data = np.reshape(np.einsum('ijk->ji',np.asarray(state_data,dtype = np.complex128)),(2**n_ion,n_num,-1))

    if 't_col' in d:
        t_col = d['t_col']
    else:
        t_col = None

    return state_data, ts, n_num, t0s, n_ion, t_col

def load_ME_seq(fname):
    
    d = np.load(fname,allow_pickle=True)
    ts = d['ts']
    s3d = d['s3d']
    n_num, n_ion = d['metadata']
    t0s = d['t0s']

    state_data = np.array([np.array(s,dtype=complex) for s in s3d], dtype= complex).reshape((-1,2**n_ion,n_num,2**n_ion,n_num))
    state_data = np.reshape(np.einsum('k...->...k',np.asarray(state_data,dtype = np.complex128)),(2**n_ion,n_num,2**n_ion,n_num,-1))

    if 't_col' in d:
        t_col = d['t_col']
    else:
        t_col = None

    return state_data, ts, n_num, t0s, n_ion, t_col

def load_QuTiP_meas(fname):
    
    d = np.load(fname,allow_pickle=True)
    ts = d['ts']
    s3d = d['s3d']
    n_num = d['metadata']
    t0s = []

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

    state_data, ts, n_num, t0s, n_ion, _ = data_pack

    state_data = np.abs(np.einsum('...,...->...',state_data,np.conj(state_data)))

    _, ax = plt.subplots()
    for i in range(n_num):
        for j in range(2**n_ion):
            atomic_state = "{0:b}".format(j+2**n_ion).replace('0','g').replace('1','e')[1:]
            if ((int(i/2) + j)%2 == 0):
                linestyle = 'dashed'
            else:
                linestyle = 'solid'
            ax.plot(1e6*ts,state_data[j,i,:],label = f"<{atomic_state},{i}|\\rho|{atomic_state},{i}>",linestyle=linestyle)

    t0 = 0
    for t in t0s:
        ax.axvline(1e6*(t0 + t),linestyle='dashdot')
        t0 += t

    ax.legend()
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])q
    # ax.get_xaxis().set_major_formatter(rabi_detuning_format)
    ax.set_xlabel("Time [$\\mu$s]")
    ax.set_ylabel("p[1]")
    # ax.set_yscale("logit")
    ax.grid()

    plt.show()

def plot_seq_scan_dm(data_pack):
    global ax, args

    state_data, ts, _, t0s, _, t_cols = data_pack
    with open(args[2]) as jsfile:
        params = json.load(jsfile)

    density_matrix = np.einsum(params['prep'],state_data,np.conj(state_data))
    _, ax = plt.subplots()
    ps = []
    legend = []
    for m_index in params['plot']:
        p, =  ax.plot(1e6*ts,np.real(density_matrix)[tuple(m_index['index'])],label = f"Re({m_index['label']})")
        ax.plot(1e6*ts,np.imag(density_matrix)[tuple(m_index['index'])],label = f"Im({m_index['label']})",linestyle='dashed',c=p.get_color())
        if(ps == []):
            ps.append(plt.Line2D([0],[0],color='black'))
            ps.append(plt.Line2D([0],[0],color='black',linestyle='dashed'))
            legend.append("Real")
            legend.append("Imag")
        ps.append(p)
        legend.append(f"{m_index['label']}")

    t0 = 0
    for t in t0s:
        ax.axvline(1e6*(t0 + t),linestyle='dashdot')
        t0 += t

    if(t_cols is not None):
        for t in t_cols:
            ax.axvline(1e6*t,0,0.25,color='red')

    ax.legend(ps,legend)
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])q
    # ax.get_xaxis().set_major_formatter(rabi_detuning_format)
    ax.set_xlabel("Time [$\\mu$s]")
    ax.set_ylabel("p[1]")
    # ax.set_yscale("logit")
    ax.grid()

    plt.show()

def plot_me_seq_scan_dm(data_pack):
    global ax, args

    state_data, ts, _, t0s, _, t_cols = data_pack
    with open(args[2]) as jsfile:
        params = json.load(jsfile)

    density_matrix = np.einsum(params['prep'],state_data)

    _, ax = plt.subplots()
    ps = []
    legend = []
    for m_index in params['plot']:
        p, =  ax.plot(1e6*ts,np.real(density_matrix)[tuple(m_index['index'])],label = f"Re({m_index['label']})")
        ax.plot(1e6*ts,np.imag(density_matrix)[tuple(m_index['index'])],label = f"Im({m_index['label']})",linestyle='dashed',c=p.get_color())
        if(ps == []):
            ps.append(plt.Line2D([0],[0],color='black'))
            ps.append(plt.Line2D([0],[0],color='black',linestyle='dashed'))
            legend.append("Real")
            legend.append("Imag")
        ps.append(p)
        legend.append(f"{m_index['label']}")

    t0 = 0
    for t in t0s:
        ax.axvline(1e6*(t0 + t),linestyle='dashdot')
        t0 += t

    if(t_cols is not None):
        for t in t_cols:
            ax.axvline(1e6*t,0,0.25,color='red')

    ax.legend(ps,legend)
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])q
    # ax.get_xaxis().set_major_formatter(rabi_detuning_format)
    ax.set_xlabel("Time [$\\mu$s]")
    ax.set_ylabel("p[1]")
    # ax.set_yscale("logit")
    ax.grid()

    plt.show()

def plot_seq_scan_projeg(data_pack):
    global ax

    state_data, ts, _, t0s, n_ion, _ = data_pack

    state_data = np.abs(np.einsum('...ik,...ik->...k',state_data,np.conj(state_data)))

    _, ax = plt.subplots()
    for i in range(2**n_ion):
        atomic_state = ("{:b}").format(i + 2**n_ion).replace('0','g').replace('1','e')[1:]
        _, = ax.plot(ts*1e6,state_data[i,:],label = f"<{atomic_state}|$\\rho$|{atomic_state}>")
    # ax.plot(ts,state_data[0,:],label = f"|g>",linestyle='--', color = p.get_color())

    t0 = 0
    for t in t0s:
        ax.axvline(1e6*(t0 + t),linestyle='dashdot')
        t0 += t

    ax.legend()
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])q
    # ax.get_xaxis().set_major_formatter(rabi_detuning_format)
    ax.set_xlabel("Time [$\\mu$s]")
    ax.set_ylabel("p[1]")
    # ax.set_yscale("logit")
    ax.grid()

    plt.show()

def plot_meas_scan(data_pack):
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
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 10))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.set_xlabel("$\\phi$ [1]")
    ax.set_ylabel("p[1]")
    # ax.set_yscale("logit")
    ax.grid()

    plt.show()

def plot_meas_scan_projeg(data_pack):
    global ax

    state_data, ts, _, t0s = data_pack

    state_data = np.abs(np.einsum('ijk,ijk->ik',state_data,np.conj(state_data)))

    _, ax = plt.subplots()
    _, = ax.plot(ts,state_data[1,:],label = f"|e>")
    # ax.plot(ts,state_data[0,:],label = f"|g>",linestyle='--', color = p.get_color())

    t0 = 0
    for t in t0s:
        ax.axvline(t0 + t,linestyle='dashdot')
        t0 += t

    ax.legend()
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])q
    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 10))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    ax.set_xlabel("$\\phi$ [1]")
    ax.set_ylabel("p[1]")
    # ax.set_yscale("logit")
    ax.grid()

    plt.show()

def plot_seq_scan_Fockexp(data_pack):
    global ax

    state_data, ts, n_num, t0s, _, _ = data_pack

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

def plot_seq_scan_Wigner(data_pack):
    global ax, fig, time_slider

    state_data, ts, n_num, t0s, _, _ = data_pack

    @lru_cache(maxsize=None)
    def coherent_state(alpha):
        ret = []
        fact = np.exp(-np.abs(alpha)**2/2)
        for i in range(n_num):
            if(i != 0):
                fact /= np.sqrt(i)
                fact *= alpha
            
            ret.append(fact)
        return np.array(ret,dtype=np.complex128)

    density_matrix = np.einsum('ijk,ilk->jlk',state_data,np.conj(state_data))
    # cat_state = (1/np.sqrt(2))*(coherent_state(5) + coherent_state(-5))
    # density_matrix = np.array(np.einsum('i,j->ij',cat_state,np.conj(cat_state))).reshape((n_num,n_num,1))

    ps = np.linspace(-3,3,301)
    xs = np.linspace(-3,3,301)

    bound = 2*const.pi/(2*(ps[1] - ps[0]))
    fft_offsets = np.linspace(-bound, bound, ps.size)

    @lru_cache(maxsize=None)
    def get_wigner_func(id):
        dmfft = []
        for x in xs:
            col = []
            for os in fft_offsets:
                col.append(np.einsum('i,ij,j->',np.conj(coherent_state(x + os/2)), density_matrix[:,:,id], coherent_state(x - os/2)))
            dmfft.append(col)
        wigner = np.abs(np.real(fftshift(fft(dmfft,axis=1),axes=1)))
        wigner = np.reshape(wigner,(xs.size,ps.size)).T
        return wigner

    wigner = get_wigner_func(0)

    fig, ax = plt.subplots()
    plt.set_cmap('gnuplot')
    fig.subplots_adjust(bottom=0.25)
    psa = np.append(ps, ps[-1] + ps[1] - ps[0])
    psa -= (ps[1] - ps[0])/2
    xsa = np.append(xs, xs[-1] + xs[1] - xs[0])
    xsa -= (xs[1] - xs[0])/2
    mesh = ax.pcolormesh(xsa,psa,wigner)
    axc = fig.colorbar(mesh)
    axc.set_label('W[$\\hbar$]')

    # ax.legend()
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])q
    # ax.get_xaxis().set_major_formatter(rabi_detuning_format)
    ax.set_xlabel("x[1]")
    ax.set_ylabel("p[$\\hbar$]")
    # ax.set_yscale("logit")
    ax.grid()

    axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(
        ax=axtime,
        label="Time[$\\mu$s]",
        valmin=ts[0]*1e6,
        valmax=ts[-1]*1e6,
        valinit=ts[0]*1e6,
        orientation="horizontal",
        valfmt="%2.4lf"
    )

    for i in t0s:
        axtime.axvline(i,linestyle='dashdot')

    def update_time(val):
        nonlocal mesh,axc
        id = np.argmin(np.abs(val - 1e6*ts))
        mesh.remove()
        wigner = get_wigner_func(id)
        mesh = ax.pcolormesh(xsa,psa,wigner)
        ax.grid()
        axc.update_normal(mesh)
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)
    plt.show()


def plot_me_seq_scan_Wigner(data_pack):
    global ax, fig, time_slider

    state_data, ts, n_num, t0s, _, _ = data_pack

    @lru_cache(maxsize=None)
    def coherent_state(alpha):
        ret = []
        fact = np.exp(-np.abs(alpha)**2/2)
        for i in range(n_num):
            if(i != 0):
                fact /= np.sqrt(i)
                fact *= alpha
            
            ret.append(fact)
        return np.array(ret,dtype=np.complex128)

    density_matrix = np.einsum('ijikm->jkm',state_data)
    # cat_state = (1/np.sqrt(2))*(coherent_state(1j))
    # density_matrix = np.array(np.einsum('i,j->ij',cat_state,np.conj(cat_state))).reshape((n_num,n_num,1))

    ps = np.linspace(-3,3,301)
    xs = np.linspace(-3,3,301)

    bound = 2*const.pi/(2*(ps[1] - ps[0]))
    fft_offsets = np.linspace(-bound, bound, ps.size)

    @lru_cache(maxsize=None)
    def get_wigner_func(id):
        dmfft = []
        for x in xs:
            col = []
            for os in fft_offsets:
                col.append(np.einsum('i,ij,j->',np.conj(coherent_state(x + os/2)), density_matrix[:,:,id], coherent_state(x - os/2)))
            dmfft.append(col)
        wigner = np.abs(np.real(fftshift(fft(dmfft,axis=1),axes=1)))
        wigner = np.reshape(wigner,(xs.size,ps.size)).T
        return wigner

    wigner = get_wigner_func(0)

    fig, ax = plt.subplots()
    plt.set_cmap('gnuplot')
    fig.subplots_adjust(bottom=0.25)
    psa = np.append(ps, ps[-1] + ps[1] - ps[0])
    psa -= (ps[1] - ps[0])/2
    xsa = np.append(xs, xs[-1] + xs[1] - xs[0])
    xsa -= (xs[1] - xs[0])/2
    mesh = ax.pcolormesh(xsa,psa,wigner)
    axc = fig.colorbar(mesh)
    axc.set_label('W[$\\hbar$]')

    # ax.legend()
    # fig2, ax2 = plt.subplots()
    # ax2.plot(t,sol[0])
    # ax2.plot(t,sol[1])q
    # ax.get_xaxis().set_major_formatter(rabi_detuning_format)
    ax.set_xlabel("x[1]")
    ax.set_ylabel("p[$\\hbar$]")
    # ax.set_yscale("logit")
    ax.grid()

    axtime = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    time_slider = Slider(
        ax=axtime,
        label="Time[$\\mu$s]",
        valmin=ts[0]*1e6,
        valmax=ts[-1]*1e6,
        valinit=ts[0]*1e6,
        orientation="horizontal",
        valfmt="%2.4lf"
    )

    for i in t0s:
        axtime.axvline(i,linestyle='dashdot')

    def update_time(val):
        nonlocal mesh,axc
        id = np.argmin(np.abs(val - 1e6*ts))
        mesh.remove()
        wigner = get_wigner_func(id)
        mesh = ax.pcolormesh(xsa,psa,wigner)
        ax.grid()
        axc.update_normal(mesh)
        fig.canvas.draw_idle()

    time_slider.on_changed(update_time)
    plt.show()


plot_methods = {
    'qutip_time'            : [load_QuTiP,plot_time_scan],
    'qutip_detuning'        : [load_QuTiP,plot_detuning_scan],
    'qutip_time_projeg'     : [load_QuTiP,plot_time_scan_projeg],
    'qutip_detuning_projeg' : [load_QuTiP,plot_detuning_scan_projeg],
    'qutip_seq'             : [load_QuTiP_seq,plot_seq_scan],
    'qutip_seq_dm'          : [load_QuTiP_seq,plot_seq_scan_dm],
    'qutip_seq_projeg'      : [load_QuTiP_seq,plot_seq_scan_projeg],
    'qutip_seq_fockexp'     : [load_QuTiP_seq,plot_seq_scan_Fockexp],
    'qutip_seq_wigner'      : [load_QuTiP_seq,plot_seq_scan_Wigner],
    'qutip_meas'            : [load_QuTiP_meas,plot_meas_scan],
    'qutip_meas_projeg'     : [load_QuTiP_meas,plot_meas_scan_projeg],
    'me_seq_dm'             : [load_ME_seq,plot_me_seq_scan_dm],
    'me_seq_wigner'         : [load_ME_seq,plot_me_seq_scan_Wigner],
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
    'font.size'         : 16
    }
    for e in rcparams.keys():
        mpl.rcParams[e] = rcparams[e]

    plt.ion()

    args = argv[1:]
    
    last = args[1]
    for m in plot_methods[args[0]]:
        last = m(last)