from ctypes import c_double, Structure, CDLL

class c_complex(Structure):
    """ creates a struct to match emxArray_real_T """

    _fields_ = [('real', c_double),
                ('imag', c_double)]

externlib = CDLL('cpp_exp.so')
exp_wrapper = externlib.exp_wrapper
exp_wrapper.argtypes = (c_double,c_double,c_double)
exp_wrapper.restype = c_complex

def c_exp(t, a, b):
    res = exp_wrapper(c_double(t),c_double(a),c_double(b))
    return res.real + 1j*res.imag

