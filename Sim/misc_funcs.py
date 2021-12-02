import qutip as qtip
import numpy as np
import scipy.constants as const
from expm_decomp import generate_qutip_exp_factor, entry, simplified_matrix_data
from functools import lru_cache

def Single_state(data):
    if(data['g']):
        state0_A = qtip.basis(2,0)
    else:
        state0_A = qtip.basis(2,1)
    state0_M = qtip.basis(data['n_num'],data['n0'])
    state0 = qtip.tensor(state0_A,state0_M)
    return state0

def Final_state(data):
    sim = np.load(data['sim_file'], allow_pickle=True)
    states = sim['s3d']
    final = qtip.Qobj(states[-1],dims=[[2,data['n_num']],[1,1]])
    return final
    
def Multiple_state(data):
    ret = qtip.Qobj(dims=[[2,data['n_num']],[1,1]])
    for s in data['states']:
        if(s['g']):
            state0_A = qtip.basis(2,0)
        else:
            state0_A = qtip.basis(2,1)
        state0_M = s['factor']*qtip.basis(data['n_num'],s['n0'])
        state0 = qtip.tensor(state0_A,state0_M)
        ret += state0
    return ret

def Coherent_state(data):
    ret = qtip.Qobj(dims=[[2,data['n_num']],[1,1]])
    e = qtip.basis(2,1)
    g = qtip.basis(2,0)
    atomic = data['e']['size']*e*np.exp(1j*const.pi*data['e']['phase']) + data['g']['size']*g*np.exp(1j*const.pi*data['g']['phase'])
    factor = 1
    for n in range(data['n_num']):
        if(n > 0):
            factor = factor/n
        basis = qtip.basis(data['n_num'],n)
        alpha = data['alpha']['size']*np.exp(1j*const.pi*data['alpha']['phase'])
        basis = basis*np.exp(-data['alpha']['size']**2/2)*alpha**n*np.sqrt(factor)
        ret += qtip.tensor(atomic,basis)
    return ret

def factor(param):
    return param['size']*np.exp(1j*param['phase']*const.pi)

def Generic_state(data):
    '''
    Generic multiparticle state builder.

    --Param structure:
    -Top level:
    states  : [state]

    -Top level but included automatically-
    n_num   : int
    n_ion   : int
    
    -state:
    n       : int
    atoms   : [boolean] / Whether given ion is in the excited state
    factor  : coefficient

    -coefficient:
    size    : float
    phase   : float / Units of pi

    '''
    ret = None
    for state in data['states']:
        state0_M = factor(state['factor'])*qtip.basis(data['n_num'],state['n'])
        state0_A = None
        for i in range(data['n_ion']):
            if(state0_A == None):
                state0_A = qtip.basis(2,1 if state['atoms'][i] else 0)
            else:
                state0_A = qtip.tensor(state0_A,qtip.basis(2,1 if state['atoms'][i] else 0))
        state0 = qtip.tensor(state0_A,state0_M)
        if(ret == None):
            ret = state0
        else:
            ret += state0
    return ret

def Generic_coherent_state(data):
    ret = None
    atomic = None
    for state in data['states']:
        state0_A = None
        for i in range(data['n_ion']):
            if(state0_A == None):
                state0_A = qtip.basis(2,1 if state['atoms'][i] else 0)
            else:
                state0_A = qtip.tensor(state0_A,qtip.basis(2,1 if state['atoms'][i] else 0))

        if(atomic == None):
            atomic = factor(state['factor'])*state0_A
        else:
            atomic += factor(state['factor'])*state0_A
    coeff = 1
    for n in range(data['n_num']):
        if(n > 0):
            coeff = coeff/n
        basis = qtip.basis(data['n_num'],n)
        alpha = data['alpha']['size']*np.exp(1j*const.pi*data['alpha']['phase'])
        basis = basis*np.exp(-data['alpha']['size']**2/2)*alpha**n*np.sqrt(coeff)
        if(ret == None):
            ret = qtip.tensor(atomic,basis)
        else:
            ret += qtip.tensor(atomic,basis)
    return ret

state_builders = {
    'Single_state'              : Single_state,
    'Final_state'               : Final_state,
    'Multiple_state'            : Multiple_state,
    'Coherent_state'            : Coherent_state,
    'Generic_state'             : Generic_state,
    'Generic_coherent_state'    : Generic_coherent_state
}
