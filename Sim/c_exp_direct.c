#include <complex.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef WIN32
#define DECLSPEC __declspec(dllexport)
#else
#define DECLSPEC
#endif

DECLSPEC static PyObject* c_exp(PyObject* self, PyObject* args){
    const double t, a;
    if(!PyArg_ParseTuple(args,"dd",&t,&a))
        return NULL;
    
    double complex val = cexp(I*t*a);
    //memcpy(&ret,&val,sizeof(Py_complex));
    // ret = {.real = creal(val), .imag = cimag(val)}
    return PyComplex_FromDoubles(creal(val),cimag(val));
}

DECLSPEC static PyMethodDef cexpMethods[] = {
    {"c_exp",  c_exp, METH_VARARGS,
     "Complex exponential of form c_exp(t : double, a : double) => e^(i*a*t)."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

DECLSPEC static struct PyModuleDef cexpmodule = {
    PyModuleDef_HEAD_INIT,
    "c_exp_direct",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    cexpMethods
};

DECLSPEC PyMODINIT_FUNC PyInit_c_exp_direct(void){
    PyObject* m;

    m = PyModule_Create(&cexpmodule);
    if(!m)
        return NULL;
    return m;
}
