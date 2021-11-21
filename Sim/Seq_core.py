#!/usr/bin/python3
import json
import numpy as np
import scipy.constants as const
from Qutip_sims import *
# from Ground_up_sims import Ground_up_full, Ground_up_LDA
import qutip as qtip
from c_exp_direct import c_exp
from copy import deepcopy

def parse_json(js_fname):
    with open(js_fname, "r") as fp:
        data = json.load(fp)
    
    if(data['nu0'] == None):
        data['nu0'] = data['nu0Hz']*2*const.pi
    
    if(data['omega0'] == None):
        data['omega0'] = 2*const.pi*data['omega0Hz']

    if(data['t_prep'] == None):
        t = 0
    else:
        t = data['t_prep']
    for d in data['sequence']:

        for i,beam in enumerate(d["beams"]):

            if(beam["Omega0"] == None):
                if(beam["Omega0Hz"] == None):
                    beam["Omega0"] = data["nu0"]*beam["Omega0_rel"]
                else:
                    beam["Omega0"] = 2*const.pi*beam["Omega0Hz"]

            if(beam["phase0"] == None):
                if(beam["phase0abs"] != None):
                    # beam["phase0"] = beam['phase0abs'] + t*(data['omega0'] + beam['detuning']*data['nu0'])
                    beam["phase0"] = np.angle(c_exp(t,beam['detuning']*data['nu0'],beam["phase0abs"]))
                    # beam["phase0"] = beam['phase0abs'] + np.angle(beam["phase0"])
                else:
                    beam["phase0"] = 0
            # if(i == 1):
            #     beam['phase0'] = d['beams'][i-1]['phase0']
        if(d["abstime"] == None):
            d['abstime'] = d["reltime"]*const.pi/d['beams'][0]['Omega0']
        
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
    npret = np.array([],dtype=object)
    for a in ret:
        for b in a:
            npret = np.append(npret,b)
    ret = npret
    if(not isinstance(ret[0],qtip.Qobj)):
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