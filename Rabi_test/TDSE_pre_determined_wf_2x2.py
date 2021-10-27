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
E_0 = 1000

# Set up the system
# The system is described, by: H_2 X H_2 x H_3
# Not all of these states exist however, such as |0,1,1> does not (1P)
# As long as everything is set up correctly, this should not pose a problem
# The mapping is such that for the 3rd index 0->0, 1->1, 2->-1.
# For higher dimensions 3->2, 4->-2, 5->3, 6->-3
# So the general term is: -(i >> 1) if i % 2 == 0 else (i >> 1) + 1
# >> -> right shift 

# These are dipole matrix elements in the x direction
dipole_m_elements = np.zeros((2,2),dtype=np.complex128)

dipole_m_elements[0,1] =  (2**7/3**5)*a_0 # <0,0, 0|x|1,1, 1> -> <1,s, 0|x|2,p, 1>
dipole_m_elements[1,0] =  (2**7/3**5)*a_0 # <0,0, 0|x|1,1,-1> -> <1,s, 0|x|2,p,-1>

# Set up energy levels
# Gross structure
E_n = lambda n : -m_e*e**4/((n+1)**2*h**3*8*c*epsilon_0**2)
# Leave forbidden levels with 0 energy
E_levels = np.array([E_n(n) for n in range(2)])

# Set up the integrator

def Schroedinger(ts, state, Hamiltonian,):
    # Reconstruct original shape
    c_state = np.reshape(state,(2,))
    # Reconstruct complex state
    # c_state = unflattened_state[0] + 1j*unflattened_state[1]

    dsdt = -(1j/hbar)*np.einsum('ij,j->i',Hamiltonian(ts),c_state)

    print(ts,end="\r")

    # Deconstruct result
    # ret = np.array([np.real(dsdt),np.imag(dsdt)])
    return dsdt.flatten()

# Set up timescale

# Set up initial state
state_0 = np.zeros((2,),dtype=np.complex128)
state_0[0] = 1

state_0_flat = state_0.flatten() # Flatten state so we can work with it

# Driving field
omega0 = (E_n(1)-E_n(0))/hbar
omega = omega0

# Rabi frequency
Omegas = np.array([[e*dipole_m_elements[i,j]*E_0 for i in range(2)] for j in range(2)])

# Set up Hamiltonian
E = np.diag([E_levels[i] for i in range(2)])

E_RWA = np.array([[[[[[-hbar*omega if n1 == n2 and l1 == l2 and m1 == m2 and n1==1 and m1 <= int(l1/2) and l1 <= n1 else 0 for m1 in range(3)] for l1 in range(2)] for n1 in range(2)] for m2 in range(3)] for l2 in range(2)] for n2 in range(2)])
Rabi_RWA = Omegas*hbar/2
def H(t):
    Rabi = np.array([[Omegas[i,j]*hbar*(np.exp(1j*omega*t) + np.exp(-1j*omega*t))/2 for i in range(2)] for j in range(2)])
    return E + Rabi

# Set up 
ts = np.arange(0,10,0.0001)
s = solve_ivp(Schroedinger,[ts[0],ts[-1]],state_0_flat,args=(H,),t_eval = ts, method='DOP853',dense_output=True,atol=1e-5,rtol=1e-5)
sol = s['y']
t = s['t']

# Recover complex results
unflattened_sol = np.reshape(sol,(2,-1))
# c_sol = unflattened_sol[0] + 1j*unflattened_sol[1]
c_sol = unflattened_sol

# Some things to make plotting more readable
states_lettering = ['s','p','d','f']
magnetic_num = lambda ml : -(ml >> 1) if ml % 2 == 0 else (ml >> 1) + 1

# Plot results
important_states =  [0,1]
fig, ax = plt.subplots()
for state in important_states:
    ax.plot(t,abs(c_sol[state])**2,label="$c_{|" + f"{state + 1}" + ">}$")

ax.legend()
plt.show()