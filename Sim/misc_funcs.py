from itertools import chain
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
    dim = []
    dim0 = []
    for _ in range(data['n_ion']):
        dim0.append(1)
        dim.append(2)
    dim0.append(1)
    dim.append(data['n_num'])
    final = qtip.Qobj(states[-1],dims=[dim,dim0])
    return final

def Final_dm(data):
    sim = np.load(data['sim_file'], allow_pickle=True)
    states = np.reshape(sim['s3d'],[-1,2**data['n_ion']*data['n_num'],2**data['n_ion']*data['n_num']])
    dim = []
    for _ in range(data['n_ion']):
        dim.append(2)
    dim.append(data['n_num'])
    final = qtip.Qobj(states[-1],dims=[dim,dim])
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

def Generic_density_matrix(data):
    state = state_builders[data['nested_builder']](data)
    return state*state.dag()

def Thermal_state(data):
    thermal = qtip.thermal_dm(data['n_num'],data['n'])
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
    return qtip.tensor(atomic*atomic.dag(),thermal)

state_builders = {
    'Single_state'              : Single_state,
    'Final_state'               : Final_state,
    'Final_dm'                  : Final_dm,
    'Multiple_state'            : Multiple_state,
    'Coherent_state'            : Coherent_state,
    'Generic_state'             : Generic_state,
    'Generic_coherent_state'    : Generic_coherent_state,
    'Generic_density_matrix'    : Generic_density_matrix,
    'Thermal_state'             : Thermal_state
}

def heating_collapse(param):
    c1 = np.sqrt(param['Gamma']*(1 + param['n_therm']))*qtip.destroy(param['n_num'])
    c2 = np.sqrt(param['Gamma']*param['n_therm'])*qtip.create(param['n_num'])

    c_A = None
    for _ in range(param['n_ion']):
        if(c_A == None):
            c_A = qtip.identity(2)
        else:
            c_A = qtip.tensor(c_A,qtip.identity(2))
        
    c1 = qtip.tensor(c_A,c1)
    c2 = qtip.tensor(c_A,c2)
    return [c1,c2]

collapse_operators = {
    'heating_collapse'      : heating_collapse
}

def raw_sequence(_,params):
    return params['sequence']

def fast_ms(data,params):
    # if (params['detuning'] is not None and params['Omega0'] is not None):
    #     detuning = params['detuning']
    #     Omega0  = params['Omega0']*data['nu0']
    #     time = const.pi*np.sqrt(params['K'])/(data['eta0']*Omega0)
    # #el
    if (params['detuning'] is not None):
        Omega0 = (params["detuning"]*data["nu0"])/(2*data['eta0']*np.sqrt(params['K']))
        #time = const.pi*np.sqrt(params['K'])/(data['eta0']*Omega0)
        detuning  = params['detuning']
    else:
        detuning = (params["Omega0"])*(2*data['eta0']*np.sqrt(params['K']))
        Omega0  = params['Omega0']*data['nu0']
    if params.get('corr',False):
        time = const.pi*np.sqrt(params['K'])/(data['eta0']*Omega0*np.exp(data['eta0']**2/2))
    else:
        time = const.pi*np.sqrt(params['K'])/(data['eta0']*Omega0)

    sequence = [{
        "reltime"           : 0,
        "abstime"           : time,
        "n_t"               : params['n_t'],
        "tau"               : params.get('tau',0),
        "beams"             : []
    }]
    sequence[0]["beams"].append({
        "Omega0"            : Omega0,
        "detuning"          : 1-detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : 0
    })
    sequence[0]["beams"].append({
        "Omega0"            : Omega0,
        "detuning"          : -1+detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : 0
    })
    return sequence

def strong_coupling2(data,params):
    r = np.roots([3*(data['eta0'])**6,-(1+data['eta0']**2)*data['eta0']**2,params['phase']])
    frac = np.sqrt(r[np.logical_and((r.imag==0),(r.real>=0))].real.min())*np.exp(0.5*data['eta0']**2)
    if (params['detuning'] is not None):
        Omega0 = -1j*frac*params['detuning']*data['nu0']
        detuning = params['detuning']
    else:
        detuning = params['Omega0']/(frac)
        Omega0 = -1j*params['Omega0']*data['nu0']

    sequence = [{
        "reltime"           : 0,
        "abstime"           : 2*const.pi/(detuning*data['nu0']),
        "n_t"               : params['n_t'],
        "tau"               : params.get('tau',0),
        "beams"             : []
    }]
    sequence[0]["beams"].append({
        "Omega0"            : -Omega0,
        "detuning"          : 1-2*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : 0
    })
    sequence[0]["beams"].append({
        "Omega0"            : Omega0,
        "detuning"          : -1+2*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : 0
    })
    sequence[0]["beams"].append({
        "Omega0"            : -Omega0,
        "detuning"          : 2-detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : 0
    })
    sequence[0]["beams"].append({
        "Omega0"            : -Omega0,
        "detuning"          : -2+detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : 0
    })
    return sequence

def strong_coupling3(data,params):
    r = np.roots([(data['eta0']**6)*382/1875,-(56/75)*(2+1/(data['eta0']**2))*data['eta0']**4,(1+2/(data['eta0']**2)+2/(data['eta0']**4))*data['eta0']**2,-5*params['phase']/(data['eta0']**4)])
    frac = np.sqrt(r[(r.imag==0) & (r.real>=0) ].real.min())
    f = frac*data['eta0']**2
    if (params['detuning'] is not None):
        Omega0 = -1j*frac*params['detuning']*data['nu0']*np.exp(0.5*data['eta0']**2)
        detuning = params['detuning']
    else:
        detuning = params['Omega0']/(frac*np.exp(0.5*data['eta0']**2))
        Omega0 = -1j*params['Omega0']*data['nu0']
    sequence = [{
        "reltime"           : 0,
        "abstime"           : 2*const.pi/(detuning*data['nu0']),
        "n_t"               : params['n_t'],
        "tau"               : params.get('tau',0),
        "beams"             : []
    }]
    sequence[0]["beams"].append({
        "Omega0"            : -Omega0,
        "detuning"          : 1-5*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : Omega0,
        "detuning"          : -1+5*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : -Omega0*2/np.sqrt(5),
        "detuning"          : 2-2*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : -Omega0*2/np.sqrt(5),
        "detuning"          : -2+2*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : -f*Omega0*7/(5),
        "detuning"          : 2+7*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : -f*Omega0*7/(5),
        "detuning"          : -2-7*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : -Omega0*np.sqrt(3/5),
        "detuning"          : 3-detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : Omega0*np.sqrt(3/5),
        "detuning"          : -3+detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    return sequence

def custom_sc2(data,params):
    r = np.roots([-data['eta0']**6*7*5/72,data['eta0']**2*(1+data['eta0']**2)*12*5/72,-params['phase']])
    frac = np.sqrt(r[(r.imag==0) & (r.real>=0) ].real.min())*np.exp(0.5*data['eta0']**2)
    params.get('j',0)
    if (params['detuning'] is not None):
        Omega0 = -1j*frac*params['detuning']*data['nu0']
        detuning = params['detuning']
    else:
        detuning = params['Omega0']/(frac)
        Omega0 = -1j*params['Omega0']*data['nu0']
    sequence = [{
        "reltime"           : 0,
        "abstime"           : 2*const.pi/(detuning*data['nu0']),
        "n_t"               : params['n_t'],
        "tau"               : params.get('tau',0),
        "beams"             : []
    }]
    sequence[0]["beams"].append({
        "Omega0"            : -Omega0,
        "detuning"          : 1-4*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : Omega0,
        "detuning"          : -1+4*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : Omega0,
        "detuning"          : 1-6*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : -Omega0,
        "detuning"          : -1+6*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : -Omega0*np.sqrt((5-2*params['j']**2)/6),
        "detuning"          : 2-1*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : -Omega0*np.sqrt((5-2*params['j']**2)/6),
        "detuning"          : -2+1*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : -params['j']*Omega0,
        "detuning"          : 2-3*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    sequence[0]["beams"].append({
        "Omega0"            : -params['j']*Omega0,
        "detuning"          : -2+3*detuning,
        "phase0abs"         : 0,
        "phase_match"       : False,
        "abspi"             : False,
        "ion"               : None,
        "phase0"            : params.get('phase0',0)
    })
    return sequence

def cardioid(data,params):
    if (params['detuning'] is not None):
        Omega0 = params['detuning']*data['nu0']/(2*data['eta0'])
        detuning = params['detuning']
    else:
        detuning = params['Omega0']*2*data['eta0']
        Omega0 = params['Omega0']*data['nu0']
    r = params['r']/np.sqrt(np.sum([ri*ri/ni for ri,ni in zip(params['r'],params['n'])]))
    sequence = [{
        "reltime"           : 0,
        "abstime"           : 2*const.pi/(detuning*data['nu0']*np.exp(data['eta0']**2/2)) if params.get('corr',False) else 2*const.pi/(detuning*data['nu0']),
        "n_t"               : params['n_t'],
        "tau"               : params.get('tau',0),
        "beams"             : []
    }]
    for ri,ni in zip(r,params['n']):
        sequence[0]["beams"].append({
            "Omega0"            : ri*Omega0,
            "detuning"          : 1-ni*detuning,
            "phase0abs"         : 0,
            "phase_match"       : False,
            "abspi"             : False,
            "ion"               : None,
            "phase0"            : 0
        })
        sequence[0]["beams"].append({
            "Omega0"            : ri*Omega0,
            "detuning"          : -1+ni*detuning,
            "phase0abs"         : 0,
            "phase_match"       : False,
            "abspi"             : False,
            "ion"               : None,
            "phase0"            : 0
        })
    return sequence

def add_carrier_S(data,params):
    sequence = sequence_builders[params['inner']['builder']](data,params['inner'])
    if (params.get('Omegac',None) is not None):
        Omega0 = params['Omegac']*data["nu0"]
    else:
        Omega0 = 2*const.pi*params['m']/sequence[0]['abstime']
    sequence[0]['beams'].append({
        "Omega0"            : Omega0,
        "carrier_corr"      : True,
        "phase0"            : 0,
        "detuning"          : 0,
        "y"                 : params['y']
    })
    return sequence

def chain_sequence(data,params):
    sequence = []
    for builder in params['seqs']:
        for pulse in sequence_builders[builder['builder']](data,builder):
            sequence.append(pulse)
    return sequence

def midway_interrupt(data,params):
    sequence = sequence_builders[params['inner']['builder']](data,params['inner'])
    return sequence

sequence_builders = {
    "raw"               : raw_sequence,
    "fast_ms"           : fast_ms,
    "strong_coupling2"  : strong_coupling2,
    "strong_coupling3"  : strong_coupling3,
    "custom_sc2"        : custom_sc2,
    "cardioid"          : cardioid,
    "Add_carrier_S"     : add_carrier_S,
    "Chain_sequence"    : chain_sequence
}