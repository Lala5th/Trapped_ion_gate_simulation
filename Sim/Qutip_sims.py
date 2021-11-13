#!/usr/bin/python3
import numpy as np
import qutip as qtip
import scipy.constants as const
from multiprocessing import Pool, Value
from expm_decomp import simplified_matrix_data, entry, generate_qutip_operator, manual_taylor_expm, generate_qutip_exp_factor
from copy import deepcopy
from cpp_exp_python import c_exp

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
    def run_sim(detuning):
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
    def run_sim(detuning):
        omega = omega0 + detuning*Omega0
        eta = z_0*omega/c
        res = qtip.sesolve(H=H_i({'omega' : omega, 'eta' : eta}),psi0=state0,tlist=ts,options=options)

        return res.states

    return run_sim
    
def QuTiP_Cython(data):

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
        ret = [[e[0],"exp(t*1j*%lf)" % (e[1],)] for e in H_i]
        return ret

    # Simulation ranges
    ts = data["ts"]

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
    def run_sim(detuning):
        omega = omega0 + detuning*Omega0
        eta = z_0*omega/c
        res = qtip.sesolve(H=H_i({'omega' : omega, 'eta' : eta}),psi0=state0,tlist=ts,options=options)

        return res.states

    return run_sim
    
def QuTiP_Cpp(data):

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
        H_i_p = [[qtip.tensor(H_A_p,H_M_p[i][0]), H_M_p[i][1]] for i in range(len(H_M_p))]
        H_i = []
        for i in H_i_p:
            H_i.append([i[0]        ,lambda t,args,e = i[1] : c_exp(t,e,arg['det'])])
            H_i.append([i[0].dag()  ,lambda t,args,e = i[1] : c_exp(t,arg['det'],e)])
        return H_i

    # Simulation ranges
    ts = data["ts"]

    # Simulation run function
    options = qtip.Options(atol=1e-8,rtol=1e-8,nsteps=1e6)
    def run_sim(detuning):
        eta = data['eta0']
        res = qtip.sesolve(H=H_i({'det' : detuning*Omega0, 'eta' : eta}),psi0=state0,tlist=ts,options=options)

        return res.states

    return run_sim
