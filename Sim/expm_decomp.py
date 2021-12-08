#!/usr/bin/python3
from dataclasses import dataclass
from typing import List,Dict,Union
from copy import deepcopy
import numpy as np
import qutip as qtip

@dataclass
class entry:
    val: np.complex128
    exp: int

    def __mul__(self, other : 'entry') -> 'entry':
        return entry(val=self.val*other.val, exp=self.exp + other.exp)

    def __div__(self, other : 'entry') -> 'entry':
        return entry(val=self.val/other.val, exp=self.exp - other.exp)

class simplified_matrix_data:
    value: List[entry]
    index: Dict[int,int]

    def rebuild_index(self) -> None:
        self.index = {}

        del_buffer = []
        for i in self.value:
            if abs(i.val) <= 1e-16:# and len(self.value) != 1:
                del_buffer.append(i)
        
        for i in del_buffer:
            self.value.remove(i)

        for i,e in enumerate(self.value):
            if (e.exp in self.index.keys()):
                self.value[self.index[e.exp]].val += self.value[i].val
                continue
            self.index[e.exp] = i

    def __repr__(self):
        return repr(self.value)


    def __str__(self):
        return str(self.value)
    def __init__(self, data :List[entry] = None) -> None:
        if(data == None):
            self.value =  [entry(0,0)]
        else:
            self.value = data
        
        self.rebuild_index()

    def __add__(self, other : 'simplified_matrix_data') -> 'simplified_matrix_data':
        new = deepcopy(self)
        
        for e in other.value:
            if (e.exp not in new.index.keys()):
                new.value.append(e)
            else:
                new.value[new.index[e.exp]].val += e.val
        
        new.rebuild_index()
        return new

    def __sub__(self, other : 'simplified_matrix_data') -> 'simplified_matrix_data':
        new = deepcopy(self)

        for e in other.value:
            if (e.exp not in new.index.keys()):
                new.value.append(entry(val = -e.value,exp = e.exp))
            else:
                new.value[new.index[e.exp]].val -= e.val
        
        new.rebuild_index()
        return new

    def mulself(self, other : 'simplified_matrix_data') -> 'simplified_matrix_data':

        data : List[entry] = []
        for e in other.value:
            for g in self.value:
                data.append(e*g)
        
        new = simplified_matrix_data(data)
        return new

    def divself(self, other : 'simplified_matrix_data') -> 'simplified_matrix_data':

        data : List[entry] = []
        for e in other.value:
            for g in self.value:
                data.append(g/e)
        
        new = simplified_matrix_data(data)
        return new

    def mulnum(self, other : Union[int,float]) -> 'simplified_matrix_data':

        new = deepcopy(self)
        for g in new.value:
            g.val *= other
        new.rebuild_index()
        return new

    def divnum(self, other : Union[int,float]) -> 'simplified_matrix_data':

        new = deepcopy(self)
        for g in new.value:
            g.val /= other
        new.rebuild_index()
        return new

    def __mul__(self, other):
        if(isinstance(other,simplified_matrix_data)):
            return self.mulself(other)
        return self.mulnum(other)

    def __truediv__(self, other):
        if(isinstance(other,simplified_matrix_data)):
            return self.divself(other)
        return self.divnum(other)

    def __lt__(self, num):
        for val in self.value:
            if(np.abs(val.val) >= num):
                return False
        return True

def manual_taylor_expm(M : np.ndarray,n : int =7) -> np.ndarray:
    ret = np.array([[simplified_matrix_data() for _ in range(M.shape[0])] for _ in range(M.shape[0])],dtype=simplified_matrix_data)
    A = np.array([[simplified_matrix_data() if i!=j else simplified_matrix_data([entry(val=1,exp=0)]) for i in range(M.shape[0])] for j in range(M.shape[1])],dtype=simplified_matrix_data)
    ret += A
    for i in range(n):
        A = A @ M
        A = A/(i+1)
        ret += A
        if (A < 1e-10).all():
            print(f"Truncating exponential at {i} as no noticeable increase is in HOT")
            break
    return ret

def generate_qutip_operator(M, exp_factor, dims = None):

    id_dict = {}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for e in M[i,j].value:
                if(e.exp not in id_dict.keys()):
                    id_dict[e.exp] = np.zeros(M.shape,dtype=np.complex)
                id_dict[e.exp][i,j] = e.val

    ret_array = []
    for i,e in enumerate(id_dict):
        ret_array.append([qtip.Qobj(id_dict[e],dims=dims), lambda t, args, ex = e : np.exp(1j*ex*exp_factor*t)])

    return ret_array

def generate_qutip_exp_factor(M, exp_factor, dims = None):
    
    id_dict = {}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for e in M[i,j].value:
                if(e.exp not in id_dict.keys()):
                    id_dict[e.exp] = np.zeros(M.shape,dtype=np.complex)
                id_dict[e.exp][i,j] = e.val

    ret_array = []
    for i,e in enumerate(id_dict):
        ret_array.append([qtip.Qobj(id_dict[e],dims=dims), e*exp_factor])

    return ret_array

def generate_python_operator(M, exp_factor, shape = None):
    if shape == None:
        shape = M.shape
    id_dict = {}
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for e in M[i,j].value:
                if(e.exp not in id_dict.keys()):
                    id_dict[e.exp] = np.zeros(M.shape,dtype=np.complex)
                id_dict[e.exp][i,j] = e.val

    for e in id_dict.keys():
        id_dict[e] = np.reshape(id_dict[e],shape)
    
    ret_array = lambda t : np.array([id_dict[e]*np.exp(1j*e*exp_factor*t) for e in id_dict.keys()])
    return lambda t : np.sum(ret_array(t),axis=0)


if __name__ == '__main__':
    n_num = 7
    a_sum = np.array([[simplified_matrix_data() for _ in range(n_num)] for _ in range(n_num)],dtype=simplified_matrix_data)
    for i in range(n_num-1):
        a_sum[i,i+1] = simplified_matrix_data([entry(val=1j*np.sqrt(i+1),exp=-1)])
        a_sum[i+1,i] = simplified_matrix_data([entry(val=1j*np.sqrt(i+1),exp= 1)])
