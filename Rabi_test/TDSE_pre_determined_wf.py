import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.constants as const

plt.ion()

# Set some constants to use later
# This makes it easier to work in natural units (h = e = c = 1)

h = 1
hbar = h/(2*const.pi)
e = 1
c = 1
epsilon_0 = e**2/(2*h*c*const.alpha)
m_e = 511e3
a_0 = 4*const.pi*epsilon_0*hbar**2/(m_e*e**2)
E_0 = 1

# Set up the system
# The system is described, by: H_2 X H_2 x H_3
# Not all of these states exist however, such as |0,1,1> does not (1P)
# As long as everything is set up correctly, this should not pose a problem
# The mapping is such that for the 3rd index 0->0, 1->1, 2->-1.
# For higher dimensions 3->2, 4->-2, 5->3, 6->-3
# So the general term is: -(i >> 1) if i % 2 == 0 else (i >> 1) + 1
# >> -> right shift 
magnetic_num = lambda ml : -(ml >> 1) if ml % 2 == 0 else (ml >> 1) + 1

# These are dipole matrix elements in the x direction
dipole_m_elements = np.zeros((2,2,3,2,2,3),dtype=np.complex128)

dipole_m_elements[0,0,0,1,1,1] =  (2**7/3**5)*a_0 # <0,0, 0|x|1,1, 1> -> <1,s, 0|x|2,p, 1>
dipole_m_elements[0,0,0,1,1,2] = -(2**7/3**5)*a_0 # <0,0, 0|x|1,1,-1> -> <1,s, 0|x|2,p,-1>
dipole_m_elements[1,1,1,0,0,0] =  (2**7/3**5)*a_0 # <1,1, 1|x|0,0, 0> -> <2,p, 1|x|1,s, 0>
dipole_m_elements[1,1,2,0,0,0] = -(2**7/3**5)*a_0 # <1,1,-1|x|0,0, 0> -> <2,p,-1|x|1,s, 0>

# Set up energy levels
# Gross structure
E_n = lambda n : -m_e*e**4/((n+1)**2*h**3*8*c*epsilon_0**2)
# Leave forbidden levels with 0 energy
E_levels = np.array([[[E_n(n) if n >= l and int(m/2) <= l else 0  for m in range(3)] for l in range(2)] for n in range(2)])

# Set up the integrator
progr = -100000
def Schroedinger(ts, state, Hamiltonian,):
    global progr
    # Reconstruct original shape
    c_state = np.reshape(state,(2,2,3))
    # Reconstruct complex state
    # c_state = unflattened_state[0] + 1j*unflattened_state[1]

    dsdt = -(1j/hbar)*np.einsum('ijklmn,lmn->ijk',Hamiltonian(ts),c_state)

    if ts - 100 > progr:
        print(ts,end="\r")
        progr = ts 
    # Deconstruct result
    # ret = np.array([np.real(dsdt),np.imag(dsdt)])
    return dsdt.flatten()

# Set up initial state
state_0 = np.zeros((2,2,3),dtype=np.complex128)
# state_0[0,0,0] = 1/np.sqrt(2)
# state_0[1,1,1] = -1/np.sqrt(2)
state_0[0,0,0] = 1

state_0_flat = state_0.flatten() # Flatten state so we can work with it


# Rabi frequency
Omegas = np.array([[[[[[e*dipole_m_elements[n1,l1,m1,n2,l2,m2]*E_0 for m1 in range(3)] for l1 in range(2)] for n1 in range(2)] for m2 in range(3)] for l2 in range(2)] for n2 in range(2)])

# Transition frequencies
omegas = np.array([[[[[[(E_levels[n1,l1,m1] - E_levels[n2,l2,m2])/hbar for m1 in range(3)] for l1 in range(2)] for n1 in range(2)] for m2 in range(3)] for l2 in range(2)] for n2 in range(2)])

# Driving field
omega0 = (E_n(1)-E_n(0))/hbar
omega = omega0 + Omegas[0,0,0,1,1,1]

# Detuning
deltas = omega - np.abs(omegas)
sums = omega + np.abs(omegas)

# Set up Hamiltonian
E = np.array([[[[[[E_levels[n1,l1,m1] if n1 == n2 and l1 == l2 and m1 == m2 and m1 <= 2*l1 and l1 <= n1 else 0 for m1 in range(3)] for l1 in range(2)] for n1 in range(2)] for m2 in range(3)] for l2 in range(2)] for n2 in range(2)])

E_RWA = np.array([[[[[[-hbar*omega if n1 == n2 and l1 == l2 and m1 == m2 and n1==1 and l1==1 and m1 <= l1*2 and l1 <= n1 else 0 for m1 in range(3)] for l1 in range(2)] for n1 in range(2)] for m2 in range(3)] for l2 in range(2)] for n2 in range(2)])
def Rabi_RWA(t):
    Hamiltonian = Omegas*hbar/2
    Hamiltonian[0,0,0,1,1,1] *= np.exp( 1j*deltas[0,0,0,1,1,1]*t)# + np.exp(-1j*sums[0,0,0,1,1,1]*t)
    Hamiltonian[0,0,0,1,1,2] *= np.exp( 1j*deltas[0,0,0,1,1,2]*t)# + np.exp(-1j*sums[0,0,0,1,1,2]*t)
    Hamiltonian[1,1,1,0,0,0] *= np.exp(-1j*deltas[1,1,1,0,0,0]*t)# + np.exp( 1j*sums[1,1,1,0,0,0]*t)
    Hamiltonian[1,1,2,0,0,0] *= np.exp(-1j*deltas[1,1,2,0,0,0]*t)# + np.exp( 1j*sums[1,1,2,0,0,0]*t)
    return Hamiltonian

def H(t):
    # Rabi = np.array([[[[[[Omegas[n1,l1,m1,n2,l2,m2]*hbar*(np.exp(1j*omega*t) + np.exp(-1j*omega*t))/2 for m1 in range(3)] for l1 in range(2)] for n1 in range(2)] for m2 in range(3)] for l2 in range(2)] for n2 in range(2)])
    # Rabi = np.zeros((2,2,3,2,2,3),dtype=np.complex128)
    # indicies = tuple(np.where(Omegas!=0))
    # field = np.exp(1j*omega*t)
    # field += np.conj(field)
    # Rabi[indicies] = Omegas[indicies]*hbar*field/2
    return Rabi_RWA(t)

# Set up solver
ts = np.arange(0,1e8,1e2)
s = solve_ivp(Schroedinger,[ts[0],ts[-1]],state_0_flat,args=(H,),t_eval = ts, method='DOP853',dense_output=True,atol=1e-9,rtol=1e-9)
sol = s['y']
t = s['t']

# Recover complex results
unflattened_sol = np.reshape(sol,(2,2,3,-1))
# c_sol = unflattened_sol[0] + 1j*unflattened_sol[1]
c_sol = unflattened_sol

# Some things to make plotting more readable
states_lettering = ['s','p','d','f']

# Plot results
important_states =  [(0,0,0),(1,1,1),(1,1,2)]
fig, ax = plt.subplots()
# prob = np.sqrt(np.abs(np.einsum('ijkl,ijkl->l',c_sol,np.conj(c_sol))))
# c_sol = c_sol / prob
for state in important_states:
    ax.plot(t,abs(c_sol[state[0],state[1],state[2]])**2,label="$c_{|" + f"{state[0] + 1},{states_lettering[state[1]]},{magnetic_num(state[2])}" + ">}$")
    # prob += abs(c_sol[state[0],state[1],state[2]])**2
prob = np.abs(np.einsum('ijkl,ijkl->l',c_sol,np.conj(c_sol)))
ax.plot(t,prob,label='Total probability')
ax.legend()
plt.show()