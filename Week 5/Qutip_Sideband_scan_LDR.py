#!/usr/bin/python3
import numpy as np
import qutip as qtip
from qutip.operators import qutrit_ops
from qutip.qobj import Qobj
from qutip.qobjevo import QobjEvo
import scipy.constants as const
from multiprocessing import Pool, Value

# Multiprocessing parameters
n_cores = 1
counter = Value('i',0)

# Set up constants
h = const.h
hbar = h/(2*const.pi)
e = const.e
c = const.c
# epsilon_0 = e**2/(2*h*c*const.alpha)
# m_e = const.m_e
# a_0 = 4*const.pi*epsilon_0*hbar**2/(m_e*e**2)

# Set up params
n_num = 7
state_start = 3
omega0 = 1e10
nu0 = 2*const.pi*1000
Omega0 = nu0/5
z_0 = 0.1*c/(omega0 + nu0)

# Set up standard operators
# Most of these could be called on demand, however 
# caching these will reduce calling overhead
a = qtip.destroy(n_num)
a_dagger = qtip.create(n_num)
sigma_p = qtip.sigmam() # Due to different convention used in previous code
sigma_m = qtip.sigmap() # and internally within QuTiP

# Create easily callable functions for modified versions of these
# Maybe later use C functions?
a_tilde = lambda t : a*np.exp(-1j*nu0*t)
# a_tilde_dagger = lambda t : a_dagger*np.exp(1j*nu0*t) # Probably calculating exponentials is harder than dag[?] Test?
a_tilde_dagger = lambda t : a_dagger*np.exp(1j*nu0*t)
def a_sum(t):
    a_t = a_tilde(t)
    return a_t + a_t.dag()
det_p = lambda t, omega : np.exp(-1j*(omega - omega0)*t)
I_A = qtip.identity(2)
I_M = qtip.identity(n_num)
identity = qtip.tensor(I_A,I_M)

# Create the initial state as the outer product H_A x H_M
state0_A = qtip.basis(2,0)
state0_M = qtip.basis(n_num,state_start)
state0 = qtip.tensor(state0_A,state0_M)

# Create Hamiltonian
# H_A_p = lambda t, args : (Omega0/2)*sigma_p*det_p(t,args['omega'])
# H_M_p = lambda t, args : (1j*args['eta']*a_sum(t)).expm()
# H_i_p = lambda t, args : qtip.tensor(H_A_p(t,args),H_M_p(t,args))
# H_i = lambda t, args : (H_i_p(t,args) + H_i_p(t,args).dag())
def H_i(args):

    d = args['omega'] - omega0

    H = []
    
    H_0 = qtip.Qobj(dims=[[2,n_num],[2,n_num]])
    H.append(H_0)

    H1_p0 = qtip.tensor(sigma_p*Omega0/2,I_M)
    H.append([H1_p0,lambda t,args : np.exp(-1j*d*t)])
    H.append([H1_p0.dag(),lambda t,args : np.exp(1j*d*t)])
    
    H1_pn = qtip.tensor(sigma_p*Omega0/2,1j*args['eta']*a)
    H.append([H1_pn,lambda t,args : np.exp(-1j*(nu0+d)*t)])
    H.append([H1_pn.dag(),lambda t,args : np.exp(1j*(nu0+d)*t)])

    H1_pn = qtip.tensor(sigma_p*Omega0/2,1j*args['eta']*a_dagger)
    H.append([H1_pn,lambda t,args : np.exp(-1j*(d-nu0)*t)])
    H.append([H1_pn.dag(),lambda t,args : np.exp(1j*(d-nu0)*t)])

    return H

# Simulation ranges
os = np.linspace(-10,10,401)
ts = np.linspace(0,10*const.pi/Omega0,2)

# Simulation run function
options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6,method='bdf')
def run_sim(detuning):
    global counter
    omega = omega0 + detuning*Omega0
    eta = z_0*omega/c
    res = qtip.sesolve(H=H_i({'omega' : omega, 'eta' : eta}),psi0=state0,tlist=ts,options=options)
    with counter.get_lock():
        counter.value += 1
    print("%2.2lf %%" % (counter.value*100/len(os)))
    return res.states # May move projection here

def init(c):
    global counter
    counter = c

if __name__ == "__main__":

    print("Done with init")
    with Pool(n_cores,initializer=init, initargs=(counter,)) as process_pool:
        result = process_pool.map(run_sim,os)
    metadata = [n_num,state_start,nu0,Omega0]
    # np.savez('a',os=os,ts=ts,metadata=metadata,s3d=np.array(result,dtype=object))

    # res = run_sim(5).states
    # proj_e = qtip.basis(2,1)*qtip.basis(2,1).dag()
    # proj_g = qtip.basis(2,0)*qtip.basis(2,0).dag()
    # fig, ax = plt.subplots()
    # for i in range(n_num):
    #     proj_M = qtip.basis(n_num,i)*qtip.basis(n_num,i).dag()
    #     proj_e_c = qtip.tensor(proj_e,proj_M)
    #     proj_g_c = qtip.tensor(proj_g,proj_M)
    #     e_as = [abs((r.dag()*proj_e_c*r)[0,0]) for r in res]
    #     g_as = [abs((r.dag()*proj_g_c*r)[0,0]) for r in res]
    #     p, = ax.plot(ts,e_as,label = f"|e,{i}>")
    #     ax.plot(ts,g_as,label = f"|g,{i}>",linestyle='--', color = p.get_color())
    #     # ax.axvline((i - state_start)*nu0/Omega0,c = p.get_color(),linestyle='dashdot')
    # ax.legend()
    # plt.show()