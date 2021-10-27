import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.linalg import expm

plt.ion()

n_num = 7

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
    c_state = np.reshape(state,(2,n_num))
    # Reconstruct complex state
    # c_state = unflattened_state[0] + 1j*unflattened_state[1]

    dsdt = -(1j/hbar)*np.einsum('ijkl,kl->ij',Hamiltonian(ts,*args),c_state)

    if ts - 1e-3 > progr:
        print(ts,end="\r")
        progr = ts 
    # Deconstruct result
    # ret = np.array([np.real(dsdt),np.imag(dsdt)])
    return dsdt.flatten()

# Set up initial state
state_0 = np.zeros((2,n_num),dtype=np.complex128)
# state_0[0,0,0] = 1/np.sqrt(2)
# state_0[1,1,1] = -1/np.sqrt(2)
state_start = 3
state_0[0,state_start] = 1

state_0_flat = state_0.flatten() # Flatten state so we can work with it

# Energy levels
E_levels = np.array([0,1e10*hbar])

# Set up Hamiltonian and relevant operators
nu0 = 1e3
H_motional = np.zeros((2,10,2,10),dtype=np.complex128)
for i in range(n_num):
    H_motional[:,i,:,i] = hbar*nu0*(i + 0.5)

a = np.zeros((n_num,n_num),dtype=np.complex128)
a_dagger = np.zeros((n_num,n_num),dtype=np.complex128)
for i in range(n_num-1):
    a[i,i+1] = np.sqrt(i+1)
    a_dagger[i+1,i] = np.sqrt(i+1)

a_tilde = lambda t : a*np.exp(-1j*nu0*t)
a_tilde_dagger = lambda t : a_dagger*np.exp(1j*nu0*t)

# Rabi frequency
Omega0 = (1/25)*nu0
Omegas = np.array([[0,Omega0],[Omega0,0]], dtype=np.complex128)
# Omegas = Omega*(sigma+ + sigma-)

E_diffs = np.array([[[[H_motional[s1,n1,s1,n1] - H_motional[s2,n2,s2,n2] + E_levels[s1] - E_levels[s2] for n2 in range(n_num)] for s2 in range(2)] for n1 in range(n_num)] for s1 in range(2)], dtype=np.complex128)

# Transition frequencies
omegas = np.array([[[[E_diffs[s1,0,s2,0]/hbar for n2 in range(n_num)] for s2 in range(2)] for n1 in range(n_num)] for s1 in range(2)])

# Driving field
omega0 = np.abs(omegas[1,0,0,0])
# omega = omega0 + Omegas[0,1]

z_0 = 10*c/(omega0 + nu0)
# Detuning
deltas = lambda omega : np.abs(abs(omega) - np.abs(omegas))
sums = lambda omega : np.abs(abs(omega) + np.abs(omegas))


proj = np.eye(n_num)
# proj[state_start,state_start] = 1
# proj[state_start-1,state_start-1] = 1
proj = np.einsum('ij,kl->ikjl',np.eye(2),proj)
# Set up Interaction Hamiltonian
def Rabi_RWA(t,omega):
    global z_0, proj
    k = omega/c
    lamb_dicke = k*z_0
    
    H_Asp = hbar*np.array([[0,0],[Omega0,0]],dtype=np.complex128)/2
    # H_Asn = hbar*np.array([[0,Omega0],[0,0]],dtype=np.complex128)/2
    ds = omega - omega0
    H_Asp *= np.exp(-1j*ds*t)
    # H_Asn *= np.exp(-1j*ds[1,0,0,0]*t)

    a_sum = a_tilde(t) + a_tilde_dagger(t)
    # mot_term = np.eye(n_num) + 1j*lamb_dicke*a_sum
    mot_term = expm( 1j*lamb_dicke*a_sum)
    # mot_term_dagger = np.conj(mot_term).T#expm(-1j*lamb_dicke*a_sum)
    #mot_term = np.einsum('ij,kl->ikjl',np.eye(2),mot_term)
    #mot_term_dagger = np.einsum('ij,kl->ikjl',np.eye(2),mot_term_dagger)

    # Hamiltonian[0,:,1,:] = mot_term[0,:,0,:] @ Hamiltonian[0,:,1,:]
    # Hamiltonian[1,:,0,:] = mot_term_dagger[1,:,1,:] @ Hamiltonian[1,:,0,:]
    # Hamiltonian[1,:,0,:] = np.conj(Hamiltonian[0,:,1,:])
    
    Hp = np.einsum('ij,kl->ikjl',H_Asp,mot_term)
    Hn = np.einsum('ijkl->klij',np.conj(Hp)) #np.einsum('ij,kl->ikjl',H_Asn,mot_term_dagger)

    Hamiltonian = Hn + Hp
    # Hamiltonian = np.einsum('ijkl,klmn->ijmn',Hamiltonian,proj)
    # Hamiltonian = np.einsum('ijkl,klmn->ijmn',proj,Hamiltonian)

    return Hamiltonian

def mod_Rabi_freq(omega):
    Ham = Rabi_RWA(0,omega)
    return 2*np.abs(Ham[0,state_start,1,state_start + int((omega - omega0)/nu0)])/hbar

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
os = np.linspace(-100,100,2001)
sidebands = np.array([(i - state_start)*nu0/Omega0 for i in range(n_num)])
for o in os:
    progr = -1000
    Omega = np.sqrt((1 + (np.min(np.abs(o - sidebands)))**2)*abs(mod_Rabi_freq(omega0 + o*Omega0))**2)
    # Omega = mod_Rabi_freq(omega0 + o*Omega0)
    ts = np.linspace(0,1*const.pi/Omega,10)
    #ts = [0,const.pi/Omega]
    s = solve_ivp(Schroedinger,[ts[0],ts[-1]],state_0_flat,args=(H,omega0 + o*Omega0),t_eval = ts, method='DOP853',dense_output=True,atol=1e-8,rtol=1e-8)
    sol = s['y']
    t = s['t']
    sol = np.einsum('ij,ij->ij',np.conj(sol),sol)
    sol = np.reshape(sol, (2,n_num,-1))
    #maxs =  sol.max(axis = 2)
    sols.append(sol[:,:,-1])
    print("                                                   ",end='\r')
    print(f"{o}")

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
for i in range(n_num):
    # col = [0.3,1,0.3]
    # if (i < state_start):
    #     col = [1 - (state_start - i - 1)/(state_start),0.2,0.2]
    # elif (i > state_start):
    #     col = [0.2,0.2,(i - state_start + 1)/(n_num - state_start)]
    p = ax.plot(os,np.abs(sols[:,1,i]),label = f"|e,{i}>")
    ax.plot(os,np.abs(sols[:,0,i]),label = f"|g,{i}>",linestyle='--', color = p[-1].get_color())
    ax.axvline((i - state_start)*nu0/Omega0,c = p[-1].get_color(),linestyle='dotted')

ax.legend()
# fig2, ax2 = plt.subplots()
# ax2.plot(t,sol[0])
# ax2.plot(t,sol[1])
# ax.get_xaxis().set_major_formatter(rabi_detuning_format)
ax.set_xlabel("Detuning [$\\Omega$]")
ax.set_ylabel("$p_\\pi$[1]")
ax.set_yscale("logit")
ax.grid()

plt.show()

# Set up solver
o = 25
Omega = np.sqrt((1 + (np.min(np.abs(o - sidebands)))**2)*abs(Omega0)**2)
Omega = mod_Rabi_freq(omega0 + o*Omega0)
ts = np.linspace(0,1*const.pi/Omega,10000)
s = solve_ivp(Schroedinger,[ts[0],ts[-1]],state_0_flat,args=(H,omega0 + o*Omega0),t_eval = ts, method='DOP853',dense_output=True,atol=1e-9,rtol=1e-9)
sol = s['y']
t = s['t']

# Recover complex results
unflattened_sol = np.reshape(sol,(2,n_num,-1))
# c_sol = unflattened_sol[0] + 1j*unflattened_sol[1]
c_sol = unflattened_sol

# Plot results
fig2, ax2 = plt.subplots()
# prob = np.sqrt(np.abs(np.einsum('ijkl,ijkl->l',c_sol,np.conj(c_sol))))
# c_sol = c_sol / prob
for i in range(n_num):
    p = ax2.plot(ts,np.abs(c_sol[1,i])**2,label = f"|e,{i}>")
    ax2.plot(ts,np.abs(c_sol[0,i])**2,label = f"|g,{i}>",linestyle='--', color = p[-1].get_color())
prob = np.abs(np.einsum('ijk,ijk->k',c_sol,np.conj(c_sol)))
ax2.plot(t,prob,label='Total probability')
ax2.legend()
ax2.grid()
plt.show()