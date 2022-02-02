#!/usr/bin/python3
import json
from matplotlib.pyplot import get
import numpy as np
import scipy.constants as const
from Qutip_sims import sim_methods, get_t_col
# from Ground_up_sims import Ground_up_full, Ground_up_LDA
import qutip as qtip
from c_exp_direct import c_exp
from expm_decomp import entry, simplified_matrix_data, manual_taylor_expm
from copy import deepcopy
from multiprocessing import Pool
from misc_funcs import sequence_builders

def get_nested_key(a,key,val):
    for k in key:
        a = a[k]
    a = val

def parse_json(js_fname):
    with open(js_fname, "r") as fp:
        js_obj = json.load(fp)

    # data = js_obj['template']
    params = js_obj['params']

    for param in params:
        param['value'] = np.linspace(*param['range'])


    return js_obj

def fill_template(data,params):
    for p in params:
        a = [data]
        for k in p['key'][:-1]:
            a.append(a[-1][k])
        a[-1][p['key'][-1]] = p['value']

    n_num = data['n_num']

    a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
    for i in range(n_num-1):
        a_sum[i,i+1] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp=-1)])
        a_sum[i+1,i] = simplified_matrix_data([entry(val=np.sqrt(i+1),exp= 1)])

    corrections = manual_taylor_expm(a_sum*1j*data['eta0'],1)
    
    if(data['nu0'] == None):
        data['nu0'] = data['nu0Hz']*2*const.pi
    
    if(data['omega0'] == None):
        data['omega0'] = 2*const.pi*data['omega0Hz']

    if(data['t_prep'] == None):
        t = 0
    else:
        t = data['t_prep']

    seq = deepcopy(data['sequence'])
    data['sequence'] = []

    for sb in seq:
        seqs = sequence_builders[sb['builder']](data,sb)
        for s in seqs:
            data['sequence'].append(s)

    for d in data['sequence']:

        for _,beam in enumerate(d["beams"]):

            if(beam["Omega0"] == None):
                if(beam["Omega0Hz"] == None):
                    beam["Omega0"] = data["nu0"]*beam["Omega0_rel"]
                else:
                    beam["Omega0"] = 2*const.pi*beam["Omega0Hz"]

            if(beam["phase0"] == None):
                if(beam["phase0abs"] != None):
                    # beam["phase0"] = beam['phase0abs'] + t*(data['omega0'] + beam['detuning']*data['nu0'])
                    beam["phase0"] = -beam["phase0abs"]
                    # beam["phase0"] = beam['phase0abs'] + np.angle(beam["phase0"])
                else:
                    beam["phase0"] = 0
            # if(i == 1):
            #     beam['phase0'] = d['beams'][i-1]['phase0']
            beam['phase0'] *= const.pi
        if(d["abstime"] == None):
            d['abstime'] = d["reltime"]*const.pi/(d['beams'][0]['Omega0']*abs(corrections[abs(int(d['beams'][0]['detuning'])),0].value[0].val))
        try:
            if(d['t0'] != None):
                t = d['t0']
        except KeyError as _:
            pass
        d['t0'] = t
        t += d['abstime']

    data['state0']['n_num'] = data['n_num']
    if('n_ion' in data):
        data['state0']['n_ion'] = data['n_ion']
    else:
        data['n_ion'] = 1
    return data
    
def run_templated_sim(data):
    
    params = data['params']

    data = fill_template(data['template'],params)

    ret = []
    t_abs = 0
    ts = np.array([])
    t_cols = np.array([])
    for i,d in enumerate(data['sequence']):
        params = deepcopy(data)
        params["beams"] = d["beams"]
        args = {}
        for j,beam in enumerate(d['beams']):
            args["det%1.0d" % (j,)] = beam['detuning']*data['nu0']
            args["phase%1.0d" % (j,)] = beam['phase0']
        params["ts"] = np.linspace(0,d["abstime"],d['n_t'])
        params['t0'] = d['t0']
        ts = np.append(ts,params['ts']+t_abs)
        t_abs = ts[-1]
        if i!= 0:
            ret.append(sim_methods[data["solver"]](params)(args,ret[-1][-1]))
        else:
            #ret.append(sim_methods[data["solver"]](params)(d['detuning']))
            ret = [sim_methods[data["solver"]](params)(args)]
        
        t_col = np.array(get_t_col())
        if(t_col != None).any():
            t_cols = np.append(t_cols, t_col + d['t0'])

        print("Completed pulse %d out of %d" % (i+1,len(data['sequence'])))
    data['ts'] = ts
    ret = np.array(ret).flatten()
    if(not isinstance(ret[0],qtip.Qobj)):
        ret = ret.reshape((ts.size,-1))

    if(t_cols.size > 0):
        data['t_col'] = np.array(t_cols).flatten()
    return ret[-1]

def run_var(js_fname):
    
    data = parse_json(js_fname)
    print("Got template:\n%s\nParams:\n%s" % (str(data['template']),str(data['params'])))

    def gen_map(params):
        ret = []
        for p in params[0]['value']:
            val = {'key' : params[0]['key'], 'value' : p}
            if(len(params) != 1):
                for a in gen_map(params[0]):
                    ret.append([a.append(val)])
            else:
                ret.append([val])     
        return ret
    ps = gen_map(data['params'])
    templates = []
    print(ps)
    for p in ps:
        templates.append(deepcopy(data))
        templates[-1]['params'] = p

    with Pool(16) as process_pool:
        results = process_pool.map(run_templated_sim,templates)

    return results,data

if __name__ == "__main__":
    __spec__ = None
    from sys import argv
    args = argv[1:]
    result, data = run_var(args[0])
    
    if(data['output']):
        if(isinstance(result[0],qtip.Qobj)):
            result = np.array(result,dtype=object)
        else:
            result = np.array(result,dtype=np.complex128)
        if(data["fname"] == None):
            data["fname"] = "temp"
        metadata = [data['template']['n_num'],data['template']['n_ion']]
        np.savez(data["fname"], s3d = result,metadata=metadata,params=[i['value'] for i in data['params']],labels=[i['label'] for i in data['params']])
