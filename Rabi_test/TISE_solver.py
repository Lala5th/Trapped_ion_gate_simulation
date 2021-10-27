import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
plt.ion()

hbar = 1

def TISE(xmin,xmax,m,spacing,potFunc,*args):
    global hbar
    xvec = np.arange(xmin,xmax,spacing) 
    Nx = len(xvec)
    V = potFunc(xvec,*args)
    second_der = (np.eye(Nx,Nx,-1) + np.eye(Nx,Nx,1) - 2*np.eye(Nx,Nx,0))/(spacing**2)
    Hamiltonian = -(hbar**2/2*m )*second_der + np.diag(V)
    print(Hamiltonian)

    eigenValues,eigenVectors = eig(Hamiltonian)
    idx = eigenValues.argsort()[::1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return eigenValues,eigenVectors,xvec
   

x_min = -10
x_max = 10
mass = 1e0
step = 0.1

QHO = lambda x, V0 : V0*x**2

Double_well = lambda x,V0,V1 : V0*x**2 + V1*x**4

two_wells = lambda x,V0,a,b,c,d : np.array([-V0 if xi < d and xi > c or xi < b and xi > a else 0 for xi in x])

inf_well = lambda x, a : np.array([0 if xi > -a and xi < a else 1e8 for xi in x])

coulomb = lambda x, Q : -Q/abs(x)

potential = QHO

#Params = [100,-2,-1,1,2]
#Params = [5,50]
Params = [5]
#Params = [0.1]

eigEnergy1i, eigFunc1i, xvec1i = TISE(x_min,x_max,mass,step,potential,*Params);

fig, ax = plt.subplots()
# ax.plot(xvec1i,10*eigFunc1i[:,0]+eigEnergy1i[0], c="b", lw=3);
# ax.plot(xvec1i,10*eigFunc1i[:,1]+eigEnergy1i[1], c="orange", lw=3)
# ax.plot(xvec1i,10*eigFunc1i[:,2]+eigEnergy1i[2], c="red", lw=3)
# ax.axhline(eigEnergy1i[0],c="b", alpha=0.5, lw=3, label="n=0")
# ax.axhline(eigEnergy1i[1], c="orange", alpha=.5, lw=3, label="n=1")
# ax.axhline(eigEnergy1i[2], c="red", alpha=.5, lw=3, label="n=2")
for i in range(10):
    p = ax.plot(xvec1i,10*eigFunc1i[:,i]+eigEnergy1i[i], lw=3)
    ax.axhline(eigEnergy1i[i], alpha=0.5, c = p[-1].get_color(), lw=3, label="n=" + str(i))
#ax_twinx = ax.twinx()
ax.plot(xvec1i,potential(xvec1i,*Params))
ax.legend()
ax.set_xlim(x_min, x_max)

from hashlib import sha256

TISE_hash = ''

with open(__file__,'rb') as f:
    bytes = f.read()
    TISE_hash = sha256(bytes).hexdigest()