import numpy as np
import matplotlib.pyplot as plt
import qutip as qtip
import scipy.constants as const
from scipy.linalg import expm
from multiprocessing import Pool, Value

plt.ion()

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
omega0 = 1e10*hbar
nu0 = 2*const.pi*1000
Omega0 = nu0/5

# Set up standard operators
# Most of these could be called on demand, however 
# caching these will reduce calling overhead
a = qtip.destroy(n_num)
a_dagger = qtip.create(n_num)
sigma_p = qtip.sigmap()
sigma_m = qtip.sigmam()

# Create easily callable functions for modified versions of these
# Maybe later use C functions?
a_tilde = lambda t : a*np.exp(-1j*nu0*t)
# a_tilde_dagger = lambda t : a_dagger*np.exp(1j*nu0*t) # Probably calculating exponentials is harder than dag[?] Test?
a_tilde_dagger = lambda t : a_tilde.dag()
a_sum = lambda t : a_tilde(t) + a_tilde_dagger(t)
det_p = lambda t, omega : np.exp(1j*(omega - omega0))

# Create the initial state as the outer product H_A x H_M
state0_A = qtip.basis(2,0)
state0_M = qtip.basis(n_num,state_start)
state0 = qtip.tensor(state0_A,state0_M)

