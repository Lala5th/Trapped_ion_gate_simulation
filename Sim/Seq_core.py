#!/usr/bin/python3
import json
import numpy as np
import scipy.constants as const
from Qutip_sims import QuTiP_C_mult_laser
# from Ground_up_sims import Ground_up_full, Ground_up_LDA
import qutip as qtip
from copy import deepcopy

sim_methods = {
    # 'Qutip_raw_func'        : None,
    # 'QuTiP_expm'            : QuTiP_full,
    # 'QuTiP_LDR'             : QuTiP_LDR,
    'QuTiP_C'               : QuTiP_C_mult_laser,
    # 'QuTiP_C'               : QuTiP_C,
    # 'Ground_up_raw_func'    : None,
    # 'Ground_up_expm'        : Ground_up_full,
    # 'Ground_up_LDR'         : Ground_up_LDA
}


def parse_json(js_fname):
    with open(js_fname, "r") as fp:
        data = json.load(fp)
    
    if(data['nu0'] == None):
        data['nu0'] = data['nu0Hz']*2*const.pi
    
    t = 0
    for d in data['sequence']:

        if(d["abstime"] == None):
            d['abstime'] = d["reltime"]*const.pi/d['beams'][0]['Omega0']
        
        for beam in d["beams"]:

            if(beam["Omega0"] == None):
                beam["Omega0"] = data["nu0"]*beam["Omega0_rel"]
            
            if(beam["phase0"] == None):
                if(beam["phase0abs"] != None):
                    beam["phase0"] = (beam['phase0abs'] + t*(data['omega0'] + beam['detuning']*data['nu0']))
                else:
                    beam["phase0"] = 0    
        t += d['abstime']
    return data

def run_sim(js_fname):
    
    data = parse_json(js_fname)

    if(data["solver"] not in sim_methods.keys()):
        raise KeyError("No solver named %s exists, use one of %s" % (data["solver"], str(sim_methods)))

    print("Got data:\n%s" % (data,))
    ret = []
    t_abs = 0
    ts = np.array([])
    for i,d in enumerate(data['sequence']):
        params = deepcopy(data)
        params["beams"] = d["beams"]
        args = {}
        for j,beam in enumerate(d['beams']):
            args["det%1.0d" % (j,)] = beam['detuning']*data['nu0']
            args["phase%1.0d" % (j,)] = beam['phase0']
        params["ts"] = np.linspace(0,d["abstime"],d['n_t'])
        ts = np.append(ts,params['ts']+t_abs)
        t_abs = ts[-1]
        if i!= 0:
            ret.append(sim_methods[data["solver"]](params)(args,ret[-1][-1]))
        else:
            #ret.append(sim_methods[data["solver"]](params)(d['detuning']))
            ret = [sim_methods[data["solver"]](params)(args)]
        print("Completed pulse %d out of %d" % (i+1,len(data['sequence'])))
    data['ts'] = ts
    ret = np.array(ret,dtype=object)
    ret = ret.reshape((-1,2*data['n_num'],1))
    return ret, data

if __name__ == "__main__":

    from sys import argv
    args = argv[1:]
    result, data = run_sim(args[0])
    
    if(data['output']):
        if(isinstance(result[0],qtip.Qobj)):
            result = np.array(result,dtype=object)
        else:
            result = np.array(result,dtype=np.complex128)
        if(data["fname"] == None):
            data["fname"] = "temp"
        metadata = [data['n_num']]
        t0s = [d['abstime'] for d in data['sequence']]
        np.savez(data["fname"], ts = data['ts'], s3d = result,metadata=metadata, t0s = t0s)