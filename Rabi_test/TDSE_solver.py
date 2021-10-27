import TISE_solver as T
import numpy as np
from importlib import reload
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import constants as const

hbar = T.hbar
e = 1
eigEnergy1i = T.eigEnergy1i
eigFunc1i = T.eigFunc1i
xvec1i = T.xvec1i
c = 1
E0 = 2.5e-1
ts = np.linspace(0,50,1001)
nmax = 2

TISE_hash_check = ''

with open('./TISE_solver.py','rb') as f:
    bytes = f.read()
    TISE_hash_check = T.sha256(bytes).hexdigest()

if(T.TISE_hash != TISE_hash_check):
    TISE_solver = reload(T)

def inner_product(a,b):
    return a.conj().dot(b)

def TDSE(s,t,H):
    sz = [s[2*i] + 1j*s[2*i+1] for i in range(int(len(s)/2))]
    dsdt = np.array(-1j*H(t).dot(sz),dtype=np.complex128)
    ret = np.array([[np.real(dsdt[i]), np.imag(dsdt[i])] for i in range(len(dsdt))]).flatten()
    return ret

omega0 = (eigEnergy1i[1] - eigEnergy1i[0])/hbar
dipole_m_elements = [[inner_product(eigFunc1i[i],xvec1i*eigFunc1i[j]) for i in range(nmax)] for j in range(nmax)]
Omega0 = e*inner_product(eigFunc1i[1],xvec1i*eigFunc1i[0])*E0/hbar
Omega = lambda t : Omega0*(np.exp(1j*omega*t) + np.exp(-1j*omega*t))
Omegas = np.array([[e*dipole_m_elements[i][j]*E0/hbar if i != j else 0 for i in range(nmax)] for j in range(nmax)])
omegas = np.array([[(eigEnergy1i[i] - eigEnergy1i[j])/hbar for i in range(nmax)] for j in range(nmax)])

omega = 1*omega0 - Omega0
omega1 = (eigEnergy1i[2] - eigEnergy1i[1])/hbar

Rabi_H_RWA = lambda t : np.array([[0,hbar*np.conj(Omega0)/2],[hbar*Omega0/2,-hbar*(omega - omega0)]],dtype=np.complex128)
Rabi_H = lambda t : np.array([[eigEnergy1i[0],hbar*np.conj(Omega(t))/2],[hbar*Omega(t)/2,eigEnergy1i[1]]],dtype=np.complex128)
N_level_Rabi_H = lambda t : np.array([[hbar*Omegas[i][j]*(np.exp(1j*omega*t) + np.exp(-1j*omega*t))/2 for i in range(nmax)] for j in range(nmax)])
N_level_Rabi_H_2 = lambda t : np.array([[hbar*Omegas[i][j]*(np.exp(1j*omega1*t) + np.exp(-1j*omega1*t))/2 for i in range(nmax)] for j in range(nmax)])
Double = lambda t : N_level_Rabi_H_2(t) + N_level_Rabi_H(t) + np.diag([eigEnergy1i[i] for i in range(nmax)])

s0 = [0,0]*nmax
s0[0] = 1
sol = odeint(TDSE,s0,ts, args=(lambda t : N_level_Rabi_H(t) + np.diag([eigEnergy1i[i] for i in range(nmax)]),))
sol2 = odeint(TDSE,[1,0,0,0],ts, args=(Rabi_H_RWA,))

fig2, ax2 = plt.subplots()
# ax2.plot(ts,(sol[:,0] + 1j*sol[:,1])*(sol[:,0] - 1j*sol[:,1]),label = 'RWA $c_0$')
# ax2.plot(ts,(sol[:,2] + 1j*sol[:,3])*(sol[:,2] - 1j*sol[:,3]),label = 'RWA $c_1$')
ax2.plot(ts,(sol2[:,0] + 1j*sol2[:,1])*(sol2[:,0] - 1j*sol2[:,1]),label ='RWA $c_0$')
ax2.plot(ts,(sol2[:,2] + 1j*sol2[:,3])*(sol2[:,2] - 1j*sol2[:,3]),label ='RWA $c_1$')
for i in range(nmax):
    ax2.plot(ts,abs(sol[:,2*i] + 1j*sol[:,2*i + 1])**2,label ='$c_'+str(i)+'$')
# ax2.plot(ts,(sol[:,2] + 1j*sol[:,3])*(sol[:,2] - 1j*sol[:,3]),label ='3-level $c_1$')
# ax2.plot(ts,(sol[:,4] + 1j*sol[:,5])*(sol[:,4] - 1j*sol[:,5]),label ='3-level $c_2$')
ax2.legend()
plt.show()