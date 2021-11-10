import numpy as np
import scipy.constants as const
import json

def parse_json(js_fname):
    with open(js_fname, "r") as fp:
        data = json.load(fp)
    
    if(data['nu0'] == None):
        data['nu0'] = data['nu0Hz']*2*const.pi
    
    if(data["Omega0"] == None):
        data["Omega0"] = data["nu0"]*data["Omega0_rel"]
    
    data["ts"] = np.linspace(0,data["max_time_rel"]*const.pi/(data["Omega0"]))

    data["os"] = np.array([np.linspace(e["min_detuning"],e["max_detuning"],e["n_det"]) for e in data["detuning_scan"]]).flatten()

    return data