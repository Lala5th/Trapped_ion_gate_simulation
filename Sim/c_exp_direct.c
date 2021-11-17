#include <complex.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject* c_exp(PyObject* self, PyObject* args){
    const double t, a, phase;
    if(!PyArg_ParseTuple(args,"ddd",&t,&a,&phase))
        return NULL;
    
    #ifdef _WIN32
        _Dcomplex val = cexp(_Cbuild(0,a*t + phase));
    #else
        double complex val = cexp(I*(t*a + phase));
    #endif
    return PyComplex_FromDoubles(creal(val),cimag(val));
}

static PyMethodDef cexpMethods[] = {
    {"c_exp",  c_exp, METH_VARARGS,
     "Complex exponential of form c_exp(t : double, a : double, phase : double) => e^(i*(a*t + phase))."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef cexpmodule = {
    PyModuleDef_HEAD_INIT,
    "c_exp_direct",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    cexpMethods
};

PyMODINIT_FUNC PyInit_c_exp_direct(void){
    PyObject* m;

    m = PyModule_Create(&cexpmodule);
    if(!m)
        return NULL;
    return m;
}
