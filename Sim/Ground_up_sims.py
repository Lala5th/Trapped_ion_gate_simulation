import numpy as np
from scipy.integrate import solve_ivp
import scipy.constants as const
from multiprocessing import Pool, Value
from expm_decomp import simplified_matrix_data, entry, manual_taylor_expm, generate_python_operator

def Ground_up_full(data):

    n_num = data["n_num"]

    # Set some constants to use later
    # Current setup uses SI units, i.e. kg, s, J, m
    c = const.c

    # Set up the ODE
    # progr = -100000
    def Schroedinger(ts, state, Hamiltonian,*args):

        # Reconstruct original shape
        c_state = np.reshape(state,(2,n_num))

        dsdt = -(1j)*np.einsum('ijkl,kl->ij',Hamiltonian(ts,*args),c_state)
        
        return dsdt.flatten()

    # Set up initial state
    state_0 = np.zeros((2,n_num),dtype=np.complex128)

    state_0[0,data['n0']] = 1

    state_0_flat = state_0.flatten() # Flatten state so we can work with it

    # Set up Hamiltonian and relevant operators
    nu0 = data['nu0']

    simga_p = np.array([[0,0],[1,0]],dtype=np.complex128)

    a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
    for i in range(n_num-1):
        a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
        a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

    # Rabi frequency
    Omega0 = data['Omega0']

    # Driving field
    omega0 = data['omega0']

    z_0 = data['eta0']*c/(omega0 + nu0)

    # Set up Interaction Hamiltonian
    def Rabi_RWA_cached(omega):
        k = omega/c
        lamb_dicke = k*z_0
        ds = omega - omega0
        
        H_Asp_0 = Omega0*simga_p/2

        H_Mp_0 = generate_python_operator(manual_taylor_expm(a_sum*1j*lamb_dicke,n=2*n_num-1),nu0)

        def Hamiltonian(t):

            H_Asp = H_Asp_0*np.exp(-1j*ds*t)
            H_Mp = H_Mp_0(t)

            Hp = np.einsum('ij,kl->ikjl',H_Asp,H_Mp)
            Hn = np.einsum('ijkl->klij',np.conj(Hp))

            return Hn + Hp

        return Hamiltonian

    # Set up solver
    ts = data['ts']

    def run_for_detuning(o):
        s = solve_ivp(Schroedinger,[ts[0],ts[-1]],state_0_flat,args=(Rabi_RWA_cached(omega0 + o*Omega0),),t_eval = ts, method='DOP853',dense_output=False,atol=1e-8,rtol=1e-8)
        sol = s['y']

        sol = sol.reshape((2,n_num,-1))

        return sol

    return run_for_detuning


def Ground_up_LDA(data):

    n_num = data["n_num"]
    # Set some constants to use later
    # Current setup uses SI units, i.e. kg, s, J, m
    c = const.c

    # Set up the ODE
    # progr = -100000
    def Schroedinger(ts, state, Hamiltonian,*args):

        # Reconstruct original shape
        c_state = np.reshape(state,(2,n_num))

        dsdt = -(1j)*np.einsum('ijkl,kl->ij',Hamiltonian(ts,*args),c_state)
        
        return dsdt.flatten()

    # Set up initial state
    state_0 = np.zeros((2,n_num),dtype=np.complex128)

    state_0[0,data['n0']] = 1

    state_0_flat = state_0.flatten() # Flatten state so we can work with it

    # Set up Hamiltonian and relevant operators
    nu0 = data['nu0']

    simga_p = np.array([[0,0],[1,0]],dtype=np.complex128)

    a = np.zeros((n_num,n_num),dtype=np.complex128)
    for i in range(n_num-1):
        a[i,i+1] = np.sqrt(i+1)

    a_tilde = lambda t : a*np.exp(-1j*nu0*t)

    # Rabi frequency
    Omega0 = data['Omega0']

    # Driving field
    omega0 = data['omega0']

    z_0 = data['eta0']*c/(omega0 + nu0)

    # Set up Interaction Hamiltonian
    I_M = np.eye(n_num)
    def Rabi_RWA_cached(omega):

        k = omega/c
        lamb_dicke = k*z_0
        ds = omega - omega0
        
        H_Asp_0 = Omega0*simga_p/2

        def Hamiltonian(t):

            H_Asp = H_Asp_0*np.exp(-1j*ds*t)

            a_sum = a_tilde(t)
            a_sum += np.conj(a_sum).T
            mot_term = I_M + 1j*lamb_dicke*a_sum

            Hp = np.einsum('ij,kl->ikjl',H_Asp,mot_term)
            Hn = np.einsum('ijkl->klij',np.conj(Hp))

            return Hn + Hp

        return Hamiltonian

    # Set up solver
    ts = data['ts']

    def run_for_detuning(o):
        s = solve_ivp(Schroedinger,[ts[0],ts[-1]],state_0_flat,args=(Rabi_RWA_cached(omega0 + o*Omega0),),t_eval = ts, method='DOP853',dense_output=False,atol=1e-8,rtol=1e-8)
        sol = s['y']

        sol = sol.reshape((2,n_num,-1))

        return sol

    return run_for_detuning