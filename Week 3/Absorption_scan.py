import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as const

plt.ion()

# Set some constants to use later
# Current setup uses SI units, i.e. kg, s, J, m

h = const.h
hbar = h/(2*const.pi)
e = const.e
c = const.c
epsilon_0 = e**2/(2*h*c*const.alpha)
m_e = const.m_e
a_0 = 4*const.pi*epsilon_0*hbar**2/(m_e*e**2)

# Set up the ODE
progr = -100000
def Schroedinger(ts, state, Hamiltonian,*args):
    global progr
    # Reconstruct original shape
    c_state = np.reshape(state,(2))
    # Reconstruct complex state
    # c_state = unflattened_state[0] + 1j*unflattened_state[1]

    dsdt = -(1j/hbar)*np.einsum('ij,j->i',Hamiltonian(ts,*args),c_state)

    if ts - 1e-3 > progr:
        print(ts,end="\r")
        progr = ts 
    # Deconstruct result
    # ret = np.array([np.real(dsdt),np.imag(dsdt)])
    return dsdt.flatten()

# Set up initial state
state_0 = np.zeros((2),dtype=np.complex128)
# state_0[0,0,0] = 1/np.sqrt(2)
# state_0[1,1,1] = -1/np.sqrt(2)
state_0[0] = 1

state_0_flat = state_0.flatten() # Flatten state so we can work with it

# Energy levels
E_levels = np.array([0,1e10*hbar])

# Rabi frequency
Omega0 = 1000000
Omegas = np.array([[0,Omega0],[np.conj(Omega0),0]], dtype=np.complex128)

# Transition frequencies
omegas = np.array([[(E_levels[j] - E_levels[i])/hbar for i in range(2)] for j in range(2)])

# Driving field
omega0 = np.abs(omegas[1,0])
# omega = omega0 + Omegas[0,1]

# Detuning
deltas = lambda omega : np.abs(abs(omega) - np.abs(omegas))
sums = lambda omega : np.abs(abs(omega) + np.abs(omegas))

# Set up Hamiltonian
def Rabi_RWA(t,omega):
    Hamiltonian = hbar*Omegas/2
    ds = deltas(omega)
    Hamiltonian[0,1] *= np.exp( 1j*ds[0,1]*t)# + np.exp(-1j*sums[0,0,0,1,1,1]*t)
    Hamiltonian[1,0] *= np.exp(-1j*ds[1,0]*t)# + np.exp( 1j*sums[1,1,1,0,0,0]*t)
    return Hamiltonian

def H(t,omega):
    # Rabi = np.array([[[[[[Omegas[n1,l1,m1,n2,l2,m2]*hbar*(np.exp(1j*omega*t) + np.exp(-1j*omega*t))/2 for m1 in range(3)] for l1 in range(2)] for n1 in range(2)] for m2 in range(3)] for l2 in range(2)] for n2 in range(2)])
    # Rabi = np.zeros((2,2,3,2,2,3),dtype=np.complex128)
    # indicies = tuple(np.where(Omegas!=0))
    # field = np.exp(1j*omega*t)
    # field += np.conj(field)
    # Rabi[indicies] = Omegas[indicies]*hbar*field/2
    return Rabi_RWA(t,omega)

# Set up solver
# ts = np.arange(0,const.pi*2/np.abs(Omegas[0,1]),1e-3*const.pi*2/np.abs(Omegas[0,1]))
sols = []
os = np.linspace(-10,10,1000)
for o in os:
    Omega = np.sqrt((1 + o**2)*abs(Omegas[0,1])**2)
    ts = [0,const.pi/Omega]
    print()
    s = solve_ivp(Schroedinger,[ts[0],ts[-1]],state_0_flat,args=(H,omega0 + o*Omegas[0,1]),t_eval = None, method='DOP853',dense_output=True,atol=1e-8,rtol=1e-8)
    sol = s['y']
    t = s['t']
    sol = np.einsum('ij,ij->ij',np.conj(sol),sol)
    sols.append(sol[:,-1])
    print(f"{o}: ",end="")

sols = np.array(sols)

# Recover complex results
# unflattened_sol = np.reshape(sol,(2,-1))
# # c_sol = unflattened_sol[0] + 1j*unflattened_sol[1]
# c_sol = unflattened_sol

# # Plot results
# fig, ax = plt.subplots()
# # prob = np.sqrt(np.abs(np.einsum('ijkl,ijkl->l',c_sol,np.conj(c_sol))))
# # c_sol = c_sol / prob
# for i in range(len(c_sol)):
#     ax.plot(t,abs(c_sol[i])**2,label="$c_{" + str(i) + "}$")
#     # prob += abs(c_sol[state[0],state[1],state[2]])**2
# prob = np.abs(np.einsum('il,il->l',c_sol,np.conj(c_sol)))
# ax.plot(t,prob,label='Total probability')
# ax.legend()
# plt.show()

rabi_detuning_format = lambda x, pos = None : f"{x} $\\Omega$" if x!=0 else "0"

fig, ax = plt.subplots()
ax.plot(os,np.abs(sols[:,1]))
# fig2, ax2 = plt.subplots()
# ax2.plot(t,sol[0])
# ax2.plot(t,sol[1])
#ax.get_xaxis().set_major_formatter(rabi_detuning_format)
ax.set_xlabel("Detuning [$\\Omega$]")
ax.set_ylabel("$p_\\pi$[1]")

plt.show()