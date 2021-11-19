import qutip as qtip
import numpy as np
import scipy.constants as const
from expm_decomp import generate_qutip_exp_factor, entry, simplified_matrix_data

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

state_builders = {
    'Single_state'      : Single_state,
    'Final_state'       : Final_state,
    'Multiple_state'    : Multiple_state
}
