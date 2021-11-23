#!/usr/bin/python3
import numpy as np
import qutip as qtip
import scipy.constants as const
from multiprocessing import Pool, Value
from expm_decomp import simplified_matrix_data, entry, generate_qutip_operator, manual_taylor_expm, generate_qutip_exp_factor
from copy import deepcopy
from c_exp_direct import c_exp
from misc_funcs import state_builders

def QuTiP_full(data):

    # Set up constants
    c = const.c

    # Set up params
    n_num = data["n_num"]
    state_start = data["n0"]
    omega0 = data['omega0']
    nu0 = data['nu0']
    Omega0 = data['Omega0']
    z_0 = data['eta0']*c/(omega0 + nu0)

    # Set up standard operators
    # Most of these could be called on demand, however 
    # caching these will reduce calling overhead
    sigma_p = qtip.sigmam() # Due to different convention used in previous code
                            # and internally within QuTiP

    # Create easily callable functions for modified versions of these
    # Maybe later use C functions?
    a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
    for i in range(n_num-1):
        a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
        a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

    # Create the initial state as the outer product H_A x H_M
    state0_A = qtip.basis(2,0)
    state0_M = qtip.basis(n_num,state_start)
    state0 = qtip.tensor(state0_A,state0_M)

    # Create Hamiltonian
    def H_i(arg):
        H_A_p = (Omega0/2)*sigma_p + 0j#*det_p(t,args['omega'])
        H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        
        # H_M_p = (1j*args['eta']*a_sum(t)).expm()
        d = arg['omega'] - omega0
        H_i_p = [[qtip.tensor(H_A_p,H_M_p[i][0]), H_M_p[i][1] - d] for i in range(len(H_M_p))]
        H_i = deepcopy(H_i_p)
        for i in H_i_p:
            H_i.append([i[0].dag(),-i[1]])
        ret = [[e[0], lambda t, args, exp = e[1] : np.exp(1j*exp*t)] for e in H_i]

        return ret

    # Simulation ranges
    ts = data["ts"]

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
    def run_sim(detuning, state0=state0):
        omega = omega0 + detuning*Omega0
        eta = z_0*omega/c
        res = qtip.sesolve(H=H_i({'omega' : omega, 'eta' : eta}),psi0=state0,tlist=ts,options=options)

        return res.states

    return run_sim

def QuTiP_LDR(data):

    # Set up constants
    c = const.c

    # Set up params
    n_num = data["n_num"]
    state_start = data["n0"]
    omega0 = data['omega0']
    nu0 = data['nu0']
    Omega0 = data['Omega0']
    z_0 = data['eta0']*c/(omega0 + nu0)

    # Set up standard operators
    # Most of these could be called on demand, however 
    # caching these will reduce calling overhead
    sigma_p = qtip.sigmam() # Due to different convention used in previous code
                            # and internally within QuTiP

    # Create easily callable functions for modified versions of these
    # Maybe later use C functions?
    a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
    for i in range(n_num-1):
        a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
        a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

    # Create the initial state as the outer product H_A x H_M
    state0_A = qtip.basis(2,0)
    state0_M = qtip.basis(n_num,state_start)
    state0 = qtip.tensor(state0_A,state0_M)

    # Create Hamiltonian
    a = qtip.destroy(n_num)
    a_dagger = qtip.create(n_num)
    I_M = qtip.identity(n_num)
    def H_i(args):
        
        d = args['omega'] - omega0

        H = []
        
        H_0 = qtip.Qobj(dims=[[2,n_num],[2,n_num]])
        H.append(H_0)

        H1_p0 = qtip.tensor(sigma_p*Omega0/2,I_M)
        H.append([H1_p0,lambda t,args : np.exp(-1j*d*t)])
        H.append([H1_p0.dag(),lambda t,args : np.exp(1j*d*t)])
        
        H1_pn = qtip.tensor(sigma_p*Omega0/2,1j*args['eta']*a)
        H.append([H1_pn,lambda t,args : np.exp(-1j*(nu0+d)*t)])
        H.append([H1_pn.dag(),lambda t,args : np.exp(1j*(nu0+d)*t)])

        H1_pn = qtip.tensor(sigma_p*Omega0/2,1j*args['eta']*a_dagger)
        H.append([H1_pn,lambda t,args : np.exp(-1j*(d-nu0)*t)])
        H.append([H1_pn.dag(),lambda t,args : np.exp(1j*(d-nu0)*t)])

        return H

    # Simulation ranges
    ts = data["ts"]

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
    def run_sim(detuning, state0=state0):
        omega = omega0 + detuning*Omega0
        eta = z_0*omega/c
        res = qtip.sesolve(H=H_i({'omega' : omega, 'eta' : eta}),psi0=state0,tlist=ts,options=options)

        return res.states

    return run_sim
    
def QuTiP_Cython(data):

    # Set up params
    n_num = data["n_num"]
    state_start = data["n0"]
    omega0 = data['omega0']
    nu0 = data['nu0']
    Omega0 = data['Omega0']

    # Set up standard operators
    # Most of these could be called on demand, however 
    # caching these will reduce calling overhead
    sigma_p = qtip.sigmam() # Due to different convention used in previous code
                            # and internally within QuTiP

    # Create easily callable functions for modified versions of these
    # Maybe later use C functions?
    a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
    for i in range(n_num-1):
        a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
        a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

    # Create the initial state as the outer product H_A x H_M
    state0_A = qtip.basis(2,0)
    state0_M = qtip.basis(n_num,state_start)
    state0 = qtip.tensor(state0_A,state0_M)

    # Create Hamiltonian
    def H_i(arg):
        H_A_p = (Omega0/2)*sigma_p + 0j#*det_p(t,args['omega'])
        H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        
        # H_M_p = (1j*args['eta']*a_sum(t)).expm()
        H_i_p = [[qtip.tensor(H_A_p,H_M_p[i][0]), H_M_p[i][1],1] for i in range(len(H_M_p))]
        H_i = deepcopy(H_i_p)
        for i in H_i_p:
            H_i.append([i[0].dag(),i[1],-1])
        ret = [[e[0],"exp(%d*t*1j*(%lf - det))" % (e[2],e[1])] for e in H_i]
        return ret

    # Simulation ranges
    ts = data["ts"]

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6,rhs_reuse=True)
    def run_sim(detuning, state0=state0):
        omega = omega0 + detuning*Omega0
        res = qtip.sesolve(H=H_i({'omega' : omega, 'eta' : data['eta0']}),args={'det' : detuning*Omega0},psi0=state0,tlist=ts,options=options)

        return res.states

    return run_sim

def QuTiP_C(data):

    # Set up params
    n_num = data["n_num"]
    state_start = data["n0"]
    nu0 = data['nu0']
    Omega0 = data['Omega0']

    # Set up standard operators
    # Most of these could be called on demand, however 
    # caching these will reduce calling overhead
    sigma_p = qtip.sigmam() # Due to different convention used in previous code
                            # and internally within QuTiP

    # Create easily callable functions for modified versions of these
    # Maybe later use C functions?
    a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
    for i in range(n_num-1):
        a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
        a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

    # Create the initial state as the outer product H_A x H_M
    state0_A = qtip.basis(2,0)
    state0_M = qtip.basis(n_num,state_start)
    state0 = qtip.tensor(state0_A,state0_M)

    # Create Hamiltonian
    def H_i(arg):
        H_A_p = (Omega0/2)*sigma_p + 0j#*det_p(t,args['omega'])
        H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        
        # H_M_p = (1j*args['eta']*a_sum(t)).expm()
        H_i_p = [[qtip.tensor(H_A_p,H_M_p[i][0]), H_M_p[i][1]] for i in range(len(H_M_p))]
        H_i = []
        for i in H_i_p:
            H_i.append([i[0]        ,lambda t,args,e = i[1] - arg['det'] : c_exp(t,e,0)])
            H_i.append([i[0].dag()  ,lambda t,args,e = arg['det'] - i[1] : c_exp(t,e,0)])
        return H_i

    # Simulation ranges
    ts = data["ts"]

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
    def run_sim(detuning, state0=state0):
        eta = data['eta0']
        res = qtip.sesolve(H=H_i({'det' : detuning*Omega0, 'eta' : eta}),psi0=state0,tlist=ts,options=options)

        return res.states

    return run_sim

def QuTiP_C_mult_laser(data):

    # Set up params
    n_num = data["n_num"]
    nu0 = data['nu0']

    # Set up standard operators
    # Most of these could be called on demand, however 
    # caching these will reduce calling overhead
    sigma_p = qtip.sigmam() # Due to different convention used in previous code
                            # and internally within QuTiP

    a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
    for i in range(n_num-1):
        a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
        a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

    # Create the initial state.
    data['state0']['n_num'] = data['n_num']
    state0 = state_builders[data['state0']['builder']](data['state0'])

    # Create Hamiltonian
    def H_i(arg):
        H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        ret = []
        for d in data['beams']:
            H_A_p = (d['Omega0']/2)*sigma_p + 0j#*det_p(t,args['omega'])
            
            # H_M_p = (1j*args['eta']*a_sum(t)).expm()
            H_i_p = [[qtip.tensor(H_A_p,H_M_p[i][0]), H_M_p[i][1],1] for i in range(len(H_M_p))]
            print(d)
            for i in H_i_p:
                ret.append([i[0]        ,lambda t,args,e = i[1] - d['detuning']*data['nu0'], b = d : c_exp(t,e, b['phase0'])])
                ret.append([i[0].dag()  ,lambda t,args,e = d['detuning']*data['nu0'] - i[1], b = d : c_exp(t,e,-b['phase0'])])
        return ret

    # Simulation ranges
    ts = data["ts"]

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
    def run_sim(args, state0=state0):
        res = qtip.sesolve(H=H_i({'eta' : data['eta0']}),psi0=state0,tlist=ts,options=options)

        return res.states

    return run_sim

def QuTiP_C_meas_mult_laser(data):
    underlying_solver = data['params']

    for beam in underlying_solver["beams"]:

        if(beam["Omega0"] == None):
            if(beam["Omega0_rel"] != None):
                beam["Omega0"] = data["nu0"]*beam["Omega0_rel"]
            else:
                beam["Omega0"] = 2*const.pi*beam["Omega0Hz"]


        if(beam["phase0"] == None):
            if(beam["phase0abs"] != None):
                beam["phase0"] = beam["phase0abs"]
            else:
                beam["phase0"] = 0    

    if(underlying_solver["abstime"] == None):
        underlying_solver['abstime'] = underlying_solver["reltime"]*const.pi/underlying_solver['beams'][0]['Omega0']
    
    underlying_solver['ts'] = np.linspace(0,underlying_solver['abstime'],underlying_solver['n_t'])

    data['state0']['n_num'] = data['n_num']

    state0 = state_builders[data['state0']['builder']](data['state0'])
    data['n0'] = 0

    # Set up params
    n_num = data['n_num']

    a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
    for i in range(n_num-1):
        a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
        a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

    # Prepare states
    p = underlying_solver
    state0 = [state0]*p['n_t']

    # Simulation run function
    def run_sim(args, state0=state0):
        data0 = deepcopy(data)
        res = []
        for j,state in enumerate(state0):
            for i,_ in enumerate(data['beams']):
                data['beams'][i]['phase0'] = np.angle(c_exp(underlying_solver['ts'][j],data['beams'][i]['detuning'],data0['beams'][i]['phase0']))
            state = qtip.Qobj(state,dims=[[2,n_num],[1,1]])
            res.append(sim_methods[data['params']['solver']](data)(args,state)[-1])
        res = np.array(res)
        return res
    return run_sim

sim_methods = {
    'QuTiP_Cython'              : QuTiP_Cython,
    'QuTiP_C_mult_laser'        : QuTiP_C_mult_laser,
    'QuTiP_C'                   : QuTiP_C,
    'QuTiP_C_meas_mult_laser'   : QuTiP_C_meas_mult_laser
}