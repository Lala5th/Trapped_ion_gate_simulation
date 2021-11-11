#!/usr/bin/python3
import json
import numpy as np
import scipy.constants as const
from Qutip_sims import QuTiP_full, QuTiP_LDR
from Ground_up_sims import Ground_up_full, Ground_up_LDA
from multiprocessing import Pool, Value
import qutip as qtip

sim_methods = {
    # 'Qutip_raw_func'        : None,
    'QuTiP_expm'            : QuTiP_full,
    'QuTiP_LDR'             : QuTiP_LDR,
    # 'Ground_up_raw_func'    : None,
    'Ground_up_expm'        : Ground_up_full,
    'Ground_up_LDR'         : Ground_up_LDA # Does not work currently, weird bug
}


def parse_json(js_fname):
    with open(js_fname, "r") as fp:
        data = json.load(fp)
    
    if(data['nu0'] == None):
        data['nu0'] = data['nu0Hz']*2*const.pi
    
    if(data["Omega0"] == None):
        data["Omega0"] = data["nu0"]*data["Omega0_rel"]
    
    data["ts"] = np.linspace(0,data["max_time_rel"]*const.pi/(data["Omega0"]),data['n_t'])

    data["os"] = np.array([np.linspace(e["min_detuning"],e["max_detuning"],e["n_det"]) for e in data["detuning_scan"]]).flatten()

    return data

counter = None
solvr = None
os = None

def init(arg, d):
    global counter, data, os
    os = d['os']
    counter = arg
    data = d

def wrap_solver(o): # Needed because pickling error
    global solvr,data
    ret = sim_methods[data["solver"]](data)(o)
    with counter.get_lock():
        counter.value += 1
    print("%2.2lf %%" % (counter.value*100/len(os)))
    return ret

def run_sim(js_fname):
    
    data = parse_json(js_fname)

    counter = Value('i',0)

    if(data["solver"] not in sim_methods.keys()):
        raise KeyError("No solver named %s exists, use one of %s" % (data["solver"], str(sim_methods)))

    print("Got data:\n%s" % (data,))

    with Pool(processes=data["n_cores"],initializer=init,initargs=(counter,data)) as proc_pool:
        print("Pool init done, starting simulation")
        ret = proc_pool.map(wrap_solver, data["os"])
    return ret, data

if __name__ == "__main__":
    
    __spec__ = None # Fixes ipython bug

    from sys import argv
    args = argv[1:]
    result, data = run_sim(args[0])
    
    if(data['output']):
        if(isinstance(result[0][0],qtip.Qobj)):
            result = np.array(result,dtype=object)
        else:
            result = np.array(result,dtype=np.complex128)
        if(data["fname"] == None):
            data["fname"] = "temp"
        metadata = [data['n_num'],data['n0'],data['nu0'],data['Omega0']]
        np.savez(data["fname"],os = data['os'], ts = data['ts'], s3d = result, metadata = metadata)