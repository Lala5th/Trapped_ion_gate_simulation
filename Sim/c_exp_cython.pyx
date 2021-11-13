#cython: language_level=3
cdef extern from "c_exp.h":
    double complex exp_wrapper(double,double,double)
c_exp = exp_wrapper