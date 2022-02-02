#!/usr/bin/python3
import numpy as np
import qutip as qtip
import scipy.constants as const
from multiprocessing import Pool, Value
from expm_decomp import simplified_matrix_data, entry, generate_qutip_operator, manual_taylor_expm, generate_qutip_exp_factor
from copy import deepcopy
from c_exp_direct import c_exp
from misc_funcs import state_builders, collapse_operators
from qutip.ui.progressbar import EnhancedTextProgressBar

def Sz(data):
    Sz = None
    sz = qtip.sigmaz()
    identity = qtip.identity(2)
    for i in range(data['n_ion']):
        Sz_p = None
        for j in range(data['n_ion']):
            c = identity if i != j else sz
            Sz_p = c if Sz_p is None else qtip.tensor(Sz_p,c)
        Sz = Sz_p if Sz is None else Sz + Sz_p
    return Sz

def Sx(data):
    Sx = None
    sx = qtip.sigmax()
    identity = qtip.identity(2)
    for i in range(data['n_ion']):
        Sx_p = None
        for j in range(data['n_ion']):
            c = identity if i != j else sx
            Sx_p = c if Sx_p is None else qtip.tensor(Sx_p,c)
        Sx = Sx_p if Sx is None else Sx + Sx_p
    return Sx

def Sy(data):
    Sy = None
    sy = qtip.sigmay()
    identity = qtip.identity(2)
    for i in range(data['n_ion']):
        Sy_p = None
        for j in range(data['n_ion']):
            c = identity if i != j else sy
            Sy_p = c if Sy_p is None else qtip.tensor(Sy_p,c)
        Sy = Sy_p if Sy is None else Sy + Sy_p
    return Sy

def state_error(data):
    return -data['xi']*qtip.tensor(Sz(data),qtip.identity(data['n_num']))/2

# def QuTiP_full(data):

#     # Set up constants
#     c = const.c

#     # Set up params
#     n_num = data["n_num"]
#     state_start = data["n0"]
#     omega0 = data['omega0']
#     nu0 = data['nu0']
#     Omega0 = data['Omega0']
#     z_0 = data['eta0']*c/(omega0 + nu0)

#     # Set up standard operators
#     # Most of these could be called on demand, however 
#     # caching these will reduce calling overhead
#     sigma_p = qtip.sigmam() # Due to different convention used in previous code
#                             # and internally within QuTiP

#     # Create easily callable functions for modified versions of these
#     # Maybe later use C functions?
#     a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
#     for i in range(n_num-1):
#         a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
#         a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

#     # Create the initial state as the outer product H_A x H_M
#     state0_A = qtip.basis(2,0)
#     state0_M = qtip.basis(n_num,state_start)
#     state0 = qtip.tensor(state0_A,state0_M)

#     # Create Hamiltonian
#     def H_i(arg):
#         H_A_p = (Omega0/2)*sigma_p + 0j#*det_p(t,args['omega'])
#         H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        
#         # H_M_p = (1j*args['eta']*a_sum(t)).expm()
#         d = arg['omega'] - omega0
#         H_i_p = [[qtip.tensor(H_A_p,H_M_p[i][0]), H_M_p[i][1] - d] for i in range(len(H_M_p))]
#         H_i = deepcopy(H_i_p)
#         for i in H_i_p:
#             H_i.append([i[0].dag(),-i[1]])
#         ret = [[e[0], lambda t, args, exp = e[1] : np.exp(1j*exp*t)] for e in H_i]

#         return ret

#     # Simulation ranges
#     ts = data["ts"]

#     # Simulation run function
#     options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
#     def run_sim(detuning, state0=state0):
#         omega = omega0 + detuning*Omega0
#         eta = z_0*omega/c
#         res = qtip.sesolve(H=H_i({'omega' : omega, 'eta' : eta}),psi0=state0,tlist=ts,options=options)

#         return res.states

#     return run_sim

# def QuTiP_LDR(data):

#     # Set up constants
#     c = const.c

#     # Set up params
#     n_num = data["n_num"]
#     state_start = data["n0"]
#     omega0 = data['omega0']
#     nu0 = data['nu0']
#     Omega0 = data['Omega0']
#     z_0 = data['eta0']*c/(omega0 + nu0)

#     # Set up standard operators
#     # Most of these could be called on demand, however 
#     # caching these will reduce calling overhead
#     sigma_p = qtip.sigmam() # Due to different convention used in previous code
#                             # and internally within QuTiP

#     # Create easily callable functions for modified versions of these
#     # Maybe later use C functions?
#     a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
#     for i in range(n_num-1):
#         a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
#         a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

#     # Create the initial state as the outer product H_A x H_M
#     state0_A = qtip.basis(2,0)
#     state0_M = qtip.basis(n_num,state_start)
#     state0 = qtip.tensor(state0_A,state0_M)

#     # Create Hamiltonian
#     a = qtip.destroy(n_num)
#     a_dagger = qtip.create(n_num)
#     I_M = qtip.identity(n_num)
#     def H_i(args):
        
#         d = args['omega'] - omega0

#         H = []
        
#         H_0 = qtip.Qobj(dims=[[2,n_num],[2,n_num]])
#         H.append(H_0)

#         H1_p0 = qtip.tensor(sigma_p*Omega0/2,I_M)
#         H.append([H1_p0,lambda t,args : np.exp(-1j*d*t)])
#         H.append([H1_p0.dag(),lambda t,args : np.exp(1j*d*t)])
        
#         H1_pn = qtip.tensor(sigma_p*Omega0/2,1j*args['eta']*a)
#         H.append([H1_pn,lambda t,args : np.exp(-1j*(nu0+d)*t)])
#         H.append([H1_pn.dag(),lambda t,args : np.exp(1j*(nu0+d)*t)])

#         H1_pn = qtip.tensor(sigma_p*Omega0/2,1j*args['eta']*a_dagger)
#         H.append([H1_pn,lambda t,args : np.exp(-1j*(d-nu0)*t)])
#         H.append([H1_pn.dag(),lambda t,args : np.exp(1j*(d-nu0)*t)])

#         return H

#     # Simulation ranges
#     ts = data["ts"]

#     # Simulation run function
#     options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
#     def run_sim(detuning, state0=state0):
#         omega = omega0 + detuning*Omega0
#         eta = z_0*omega/c
#         res = qtip.sesolve(H=H_i({'omega' : omega, 'eta' : eta}),psi0=state0,tlist=ts,options=options)

#         return res.states

#     return run_sim
    
# def QuTiP_Cython(data):

#     # Set up params
#     n_num = data["n_num"]
#     state_start = data["n0"]
#     omega0 = data['omega0']
#     nu0 = data['nu0']
#     Omega0 = data['Omega0']

#     # Set up standard operators
#     # Most of these could be called on demand, however 
#     # caching these will reduce calling overhead
#     sigma_p = qtip.sigmam() # Due to different convention used in previous code
#                             # and internally within QuTiP

#     # Create easily callable functions for modified versions of these
#     # Maybe later use C functions?
#     a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
#     for i in range(n_num-1):
#         a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
#         a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

#     # Create the initial state as the outer product H_A x H_M
#     state0_A = qtip.basis(2,0)
#     state0_M = qtip.basis(n_num,state_start)
#     state0 = qtip.tensor(state0_A,state0_M)

#     # Create Hamiltonian
#     def H_i(arg):
#         H_A_p = (Omega0/2)*sigma_p + 0j#*det_p(t,args['omega'])
#         H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        
#         # H_M_p = (1j*args['eta']*a_sum(t)).expm()
#         H_i_p = [[qtip.tensor(H_A_p,H_M_p[i][0]), H_M_p[i][1],1] for i in range(len(H_M_p))]
#         H_i = deepcopy(H_i_p)
#         for i in H_i_p:
#             H_i.append([i[0].dag(),i[1],-1])
#         ret = [[e[0],"exp(%d*t*1j*(%lf - det))" % (e[2],e[1])] for e in H_i]
#         return ret

#     # Simulation ranges
#     ts = data["ts"]

#     # Simulation run function
#     options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6,rhs_reuse=True)
#     def run_sim(detuning, state0=state0):
#         omega = omega0 + detuning*Omega0
#         res = qtip.sesolve(H=H_i({'omega' : omega, 'eta' : data['eta0']}),args={'det' : detuning*Omega0},psi0=state0,tlist=ts,options=options)

#         return res.states

#     return run_sim

# def QuTiP_C(data):

#     # Set up params
#     n_num = data["n_num"]
#     state_start = data["n0"]
#     nu0 = data['nu0']
#     Omega0 = data['Omega0']

#     # Set up standard operators
#     # Most of these could be called on demand, however 
#     # caching these will reduce calling overhead
#     sigma_p = qtip.sigmam() # Due to different convention used in previous code
#                             # and internally within QuTiP

#     # Create easily callable functions for modified versions of these
#     # Maybe later use C functions?
#     a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
#     for i in range(n_num-1):
#         a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
#         a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

#     # Create the initial state as the outer product H_A x H_M
#     state0_A = qtip.basis(2,0)
#     state0_M = qtip.basis(n_num,state_start)
#     state0 = qtip.tensor(state0_A,state0_M)

#     # Create Hamiltonian
#     def H_i(arg):
#         H_A_p = (Omega0/2)*sigma_p + 0j#*det_p(t,args['omega'])
#         H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        
#         # H_M_p = (1j*args['eta']*a_sum(t)).expm()
#         H_i_p = [[qtip.tensor(H_A_p,H_M_p[i][0]), H_M_p[i][1]] for i in range(len(H_M_p))]
#         H_i = []
#         for i in H_i_p:
#             H_i.append([i[0]        ,lambda t,args,e = i[1] - arg['det'] : c_exp(t,e,0)])
#             H_i.append([i[0].dag()  ,lambda t,args,e = arg['det'] - i[1] : c_exp(t,e,0)])
#         return H_i

#     # Simulation ranges
#     ts = data["ts"]

#     # Simulation run function
#     options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
#     def run_sim(detuning, state0=state0):
#         eta = data['eta0']
#         res = qtip.sesolve(H=H_i({'det' : detuning*Omega0, 'eta' : eta}),psi0=state0,tlist=ts,options=options)

#         return res.states

#     return run_sim

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
                ret.append([i[0]        ,lambda t,args,e = i[1] - d['detuning']*data['nu0'], b = d : c_exp(t + data['t0'],e, b['phase0'])])
                ret.append([i[0].dag()  ,lambda t,args,e = d['detuning']*data['nu0'] - i[1], b = d : c_exp(t + data['t0'],e,-b['phase0'])])
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
                data['beams'][i]['phase0'] = underlying_solver['ts'][j]*data['beams'][i]['detuning'] + data0['beams'][i]['phase0']
            state = qtip.Qobj(state,dims=[[2,n_num],[1,1]])
            res.append(sim_methods[data['params']['solver']](data)(args,state)[-1])
        res = np.array(res)
        return res
    return run_sim

def QuTiP_C_mult_laser_generic(data):

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
    state0 = state_builders[data['state0']['builder']](data['state0'])

    # Create Hamiltonian
    def H_i(arg):
        H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        ret = []
        for d in data['beams']:
            H_A_p = (d['Omega0']/2)*sigma_p + 0j#*det_p(t,args['omega'])
            
            # H_M_p = (1j*args['eta']*a_sum(t)).expm()
            address = None
            if("ion" in d):
                address = d['ion']
            H_i_p = []
            for i in range(len(H_M_p)):
                H_data = None
                for j in range(data['n_ion']):
                    H_p = None
                    if(address!=None):
                        if(address!=j):
                            continue
                    for k in range(data['n_ion']):
                        H_part = qtip.identity(2) if j!=k else H_A_p
                        if(H_p == None):
                            H_p = H_part
                        else:
                            H_p = qtip.tensor(H_p,H_part)
                    if(H_data == None):
                        H_data = H_p
                    else:
                        H_data += H_p
                H_i_p.append([qtip.tensor(H_data,H_M_p[i][0]), H_M_p[i][1],1])
                
            for i in H_i_p:
                ret.append([i[0]        ,lambda t,args,e = i[1] - d['detuning']*data['nu0'], b = d : c_exp(t + data['t0'],e, b['phase0'])])
                ret.append([i[0].dag()  ,lambda t,args,e = d['detuning']*data['nu0'] - i[1], b = d : c_exp(t + data['t0'],e,-b['phase0'])])
        return ret

    # Simulation ranges
    ts = data["ts"]

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
    def run_sim(args, state0=state0):
        res = qtip.sesolve(H=H_i({'eta' : data['eta0']}),psi0=state0,tlist=ts,options=options)

        return res.states

    return run_sim

t_col = None

def get_t_col():
    global t_col
    temp = deepcopy(t_col)
    t_col = None
    return temp

def QuTiP_C_mult_laser_generic_collapse(data):
    global t_col

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
    state0 = state_builders[data['state0']['builder']](data['state0'])

    # Create Hamiltonian
    def H_i(arg):
        H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        ret = []
        for d in data['beams']:
            H_A_p = (d['Omega0']/2)*sigma_p + 0j#*det_p(t,args['omega'])
            
            # H_M_p = (1j*args['eta']*a_sum(t)).expm()
            address = None
            if("ion" in d):
                address = d['ion']
            H_i_p = []
            for i in range(len(H_M_p)):
                H_data = None
                for j in range(data['n_ion']):
                    H_p = None
                    if(address!=None):
                        if(address!=j):
                            continue
                    for k in range(data['n_ion']):
                        H_part = qtip.identity(2) if j!=k else H_A_p
                        if(H_p == None):
                            H_p = H_part
                        else:
                            H_p = qtip.tensor(H_p,H_part)
                    if(H_data == None):
                        H_data = H_p
                    else:
                        H_data += H_p
                H_i_p.append([qtip.tensor(H_data,H_M_p[i][0]), H_M_p[i][1],1])
                
            for i in H_i_p:
                ret.append([i[0]        ,lambda t,args,e = i[1] - d['detuning']*data['nu0'], b = d : c_exp(t + data['t0'],e, b['phase0'])])
                ret.append([i[0].dag()  ,lambda t,args,e = d['detuning']*data['nu0'] - i[1], b = d : c_exp(t + data['t0'],e,-b['phase0'])])
        err = state_error(data)
        if err is not None:
            ret.append(err)
        return ret

    params = data['c_param']
    params['n_ion'] = data['n_ion']
    params['n_num'] = data['n_num']
    c_ops = collapse_operators[params['c_operator']](params)

    # Simulation ranges
    ts = data["ts"]

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
    def run_sim(args, state0=state0):
        global t_col
        # print(state0.shape)
        res = qtip.mcsolve(H=H_i({'eta' : data['eta0']}),psi0=state0,tlist=ts,options=options,c_ops=c_ops,map_func=qtip.serial_map,ntraj=data['ntraj'],progress_bar=EnhancedTextProgressBar())
        t_col = res.col_times

        return res.states[0]

    return run_sim

def ME_C_mult_laser_generic_collapse(data):
    global t_col

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
    state0 = state_builders[data['state0']['builder']](data['state0'])

    # Create Hamiltonian
    def H_i(arg):
        H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        ret = []
        for d in data['beams']:
            H_A_p = (d['Omega0']/2)*sigma_p + 0j#*det_p(t,args['omega'])
            
            # H_M_p = (1j*args['eta']*a_sum(t)).expm()
            address = None
            if("ion" in d):
                address = d['ion']
            H_i_p = []
            for i in range(len(H_M_p)):
                H_data = None
                for j in range(data['n_ion']):
                    H_p = None
                    if(address!=None):
                        if(address!=j):
                            continue
                    for k in range(data['n_ion']):
                        H_part = qtip.identity(2) if j!=k else H_A_p
                        if(H_p == None):
                            H_p = H_part
                        else:
                            H_p = qtip.tensor(H_p,H_part)
                    if(H_data == None):
                        H_data = H_p
                    else:
                        H_data += H_p
                H_i_p.append([qtip.tensor(H_data,H_M_p[i][0]), H_M_p[i][1],1])
                
            for i in H_i_p:
                ret.append([i[0]        ,lambda t,args,e = i[1] - d['detuning']*data['nu0'], b = d : c_exp(t + data['t0'],e, b['phase0'])])
                ret.append([i[0].dag()  ,lambda t,args,e = d['detuning']*data['nu0'] - i[1], b = d : c_exp(t + data['t0'],e,-b['phase0'])])
        err = state_error(data)
        if err is not None:
            ret.append(err)
        return ret

    params = data['c_param']
    params['n_ion'] = data['n_ion']
    params['n_num'] = data['n_num']
    c_ops = collapse_operators[params['c_operator']](params)

    # Simulation ranges
    ts = data["ts"]

    state0 = state0

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
    def run_sim(args, state0=state0):
        global t_col
        # print(state0.shape)
        res = qtip.mesolve(H=H_i({'eta' : data['eta0']}),rho0=state0,tlist=ts,options=options,c_ops=c_ops,progress_bar=EnhancedTextProgressBar())
        return res.states

    return run_sim

def SC_paper(data):
    global t_col

    # Set up params
    n_num = data["n_num"]
    nu0 = data['nu0']

    # Set up standard operators
    # Most of these could be called on demand, however 
    # caching these will reduce calling overhead
    sigma_p = qtip.sigmam() # Due to different convention used in previous code
                            # and internally within QuTiP

    a = np.zeros((n_num,n_num))
    ad = np.zeros((n_num,n_num))
    for i in range(n_num-1):
        a[i,i+1] = np.sqrt(i+1)
        ad[i+1,i] = np.sqrt(i+1)

    def get_diagonals():
        ret = []
        for k in range(n_num):
            ret.append([qtip.Qobj(np.exp(-data['eta0']**2/2)*np.sum([(1j*data['eta0'])**(2*i + k)*(np.linalg.matrix_power(ad,i + k)@np.linalg.matrix_power(a,i))/(np.math.factorial(i+k)*np.math.factorial(i)) for i in range(2*n_num)],axis=0)),k*data['nu0']])
        for k in range(1,n_num):
            ret.append([qtip.Qobj(np.exp(-data['eta0']**2/2)*np.sum([(1j*(-data['eta0']))**(2*i + k)*(np.linalg.matrix_power(ad,i + k)@np.linalg.matrix_power(a,i))/(np.math.factorial(i+k)*np.math.factorial(i)) for i in range(2*n_num)],axis=0)).dag(),-k*data['nu0']])
        return ret

    # Create the initial state.
    state0 = state_builders[data['state0']['builder']](data['state0'])

    a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
    for i in range(n_num-1):
        a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
        a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

    # Create the initial state.
    state0 = state_builders[data['state0']['builder']](data['state0'])

    # Create Hamiltonian
    def H_i(arg):
        H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        ret = []
        for d in data['beams']:
            H_A_p = (d['Omega0']/2)*sigma_p + 0j#*det_p(t,args['omega'])
            
            # H_M_p = (1j*args['eta']*a_sum(t)).expm()
            address = None
            if("ion" in d):
                address = d['ion']
            H_i_p = []
            for i in range(len(H_M_p)):
                H_data = None
                for j in range(data['n_ion']):
                    H_p = None
                    if(address!=None):
                        if(address!=j):
                            continue
                    for k in range(data['n_ion']):
                        H_part = qtip.identity(2) if j!=k else H_A_p
                        if(H_p == None):
                            H_p = H_part
                        else:
                            H_p = qtip.tensor(H_p,H_part)
                    if(H_data == None):
                        H_data = H_p
                    else:
                        H_data += H_p
                H_i_p.append([qtip.tensor(H_data,H_M_p[i][0]), H_M_p[i][1],1])
                
            for i in H_i_p:
                if np.abs(i[1]/data['nu0'] - d['detuning']) > 0.5:
                    continue
                ret.append([i[0]        ,lambda t,args,e = i[1] - d['detuning']*data['nu0'], b = d : c_exp(t + data['t0'],e, b['phase0'])])
                ret.append([i[0].dag()  ,lambda t,args,e = d['detuning']*data['nu0'] - i[1], b = d : c_exp(t + data['t0'],e,-b['phase0'])])
        err = state_error(data)
        if err is not None:
            ret.append(err)
        return ret

    params = data['c_param']
    params['n_ion'] = data['n_ion']
    params['n_num'] = data['n_num']
    c_ops = collapse_operators[params['c_operator']](params)

    # Simulation ranges
    ts = data["ts"]

    state0 = state0

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
    def run_sim(args, state0=state0):
        global t_col
        # print(state0.shape)
        res = qtip.mesolve(H=H_i({'eta' : data['eta0']}),rho0=state0,tlist=ts,options=options,c_ops=c_ops,progress_bar=EnhancedTextProgressBar())
        return res.states

    return run_sim

def ME_C_mult_laser_generic_collapse_reduced(data):
    global t_col

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
    state0 = state_builders[data['state0']['builder']](data['state0'])

    # Create Hamiltonian
    def H_i(arg):
        H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        ret = []
        for d in data['beams']:
            H_A_p = (d['Omega0']/2)*sigma_p + 0j#*det_p(t,args['omega'])
            
            # H_M_p = (1j*args['eta']*a_sum(t)).expm()
            address = None
            if("ion" in d):
                address = d['ion']
            H_i_p = []
            for i in range(len(H_M_p)):
                H_data = None
                for j in range(data['n_ion']):
                    H_p = None
                    if(address!=None):
                        if(address!=j):
                            continue
                    for k in range(data['n_ion']):
                        H_part = qtip.identity(2) if j!=k else H_A_p
                        if(H_p == None):
                            H_p = H_part
                        else:
                            H_p = qtip.tensor(H_p,H_part)
                    if(H_data == None):
                        H_data = H_p
                    else:
                        H_data += H_p
                H_i_p.append([qtip.tensor(H_data,H_M_p[i][0]), H_M_p[i][1],1])
                
            for i in H_i_p:
                if abs(i[1]/data['nu0'] - d['detuning']) > np.abs(20*d['Omega0']/data['nu0']):
                    continue
                ret.append([i[0]        ,lambda t,args,e = i[1] - d['detuning']*data['nu0'], b = d : c_exp(t + data['t0'],e, b['phase0'])])
                ret.append([i[0].dag()  ,lambda t,args,e = d['detuning']*data['nu0'] - i[1], b = d : c_exp(t + data['t0'],e,-b['phase0'])])
        err = state_error(data)
        if err is not None:
            ret.append(err)
        return ret

    params = data['c_param']
    params['n_ion'] = data['n_ion']
    params['n_num'] = data['n_num']
    c_ops = collapse_operators[params['c_operator']](params)

    # Simulation ranges
    ts = data["ts"]

    state0 = state0

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
    def run_sim(args, state0=state0):
        global t_col
        # print(state0.shape)
        res = qtip.mesolve(H=H_i({'eta' : data['eta0']}),rho0=state0,tlist=ts,options=options,c_ops=c_ops,progress_bar=EnhancedTextProgressBar())
        return res.states

    return run_sim

def ME_Interaction_OR(data):
    global t_col

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
    state0 = state_builders[data['state0']['builder']](data['state0'])

    # Create Hamiltonian
    def H_i(arg):
        C_corr = False
        H_M_p = generate_qutip_exp_factor(manual_taylor_expm(a_sum*1j*arg['eta'],n=2*n_num), nu0)
        ret = []
        for d in data['beams']:

            if(d.get('carrier_corr',False)):
                ret.append([-0.5*data['xi']*qtip.tensor(Sz(data),qtip.identity(data['n_num'])),lambda t, _ : np.cos(np.abs(d['Omega0'])*t)])
                ret.append([-0.5*data['xi']*qtip.tensor(Sy(data) if d['y'] else Sx(data),qtip.identity(data['n_num'])),lambda t, _ : np.sin(np.abs(d['Omega0'])*t)])
                assert not C_corr
                C_corr = True
                continue

            H_A_p = (d['Omega0']/2)*sigma_p + 0j#*det_p(t,args['omega'])
            
            # H_M_p = (1j*args['eta']*a_sum(t)).expm()
            address = None
            if("ion" in d):
                address = d['ion']
            H_i_p = []
            for i in range(len(H_M_p)):
                H_data = None
                for j in range(data['n_ion']):
                    H_p = None
                    if(address!=None):
                        if(address!=j):
                            continue
                    for k in range(data['n_ion']):
                        H_part = qtip.identity(2) if j!=k else H_A_p
                        if(H_p == None):
                            H_p = H_part
                        else:
                            H_p = qtip.tensor(H_p,H_part)
                    if(H_data == None):
                        H_data = H_p
                    else:
                        H_data += H_p
                H_i_p.append([qtip.tensor(H_data,H_M_p[i][0]), H_M_p[i][1],1])
                
            for i in H_i_p:
                if abs(i[1]/data['nu0'] - d['detuning']) > np.abs(20*d['Omega0']/data['nu0']):
                    continue
                ret.append([i[0]        ,lambda t,args,e = i[1] - d['detuning']*data['nu0'], b = d : c_exp(t + data['t0'],e, b['phase0'])])
                ret.append([i[0].dag()  ,lambda t,args,e = d['detuning']*data['nu0'] - i[1], b = d : c_exp(t + data['t0'],e,-b['phase0'])])
        err = state_error(data)
        if err is not None and not C_corr:
            ret.append(err)
        return ret

    params = data['c_param']
    params['n_ion'] = data['n_ion']
    params['n_num'] = data['n_num']
    c_ops = collapse_operators[params['c_operator']](params)

    # Simulation ranges
    ts = data["ts"]

    state0 = state0

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
    def run_sim(args, state0=state0):
        global t_col
        # print(state0.shape)
        res = qtip.mesolve(H=H_i({'eta' : data['eta0']}),rho0=state0,tlist=ts,options=options,c_ops=c_ops,progress_bar=EnhancedTextProgressBar())
        return res.states

    return run_sim

sim_methods = {
    # 'QuTiP_Cython'                          : QuTiP_Cython,
    'QuTiP_C_mult_laser'                        : QuTiP_C_mult_laser,
    # 'QuTiP_C'                               : QuTiP_C,
    'QuTiP_C_meas_mult_laser'                   : QuTiP_C_meas_mult_laser,
    'QuTiP_C_mult_laser_generic'                : QuTiP_C_mult_laser_generic,
    'QuTiP_C_mult_laser_generic_collapse'       : QuTiP_C_mult_laser_generic_collapse,
    'ME_C_mult_laser_generic_collapse'          : ME_C_mult_laser_generic_collapse,
    'ME_C_mult_laser_generic_collapse_reduced'  : ME_C_mult_laser_generic_collapse_reduced,
    'SC_paper'                                  : SC_paper,
    'ME_Interaction_OR'                         : ME_Interaction_OR
}