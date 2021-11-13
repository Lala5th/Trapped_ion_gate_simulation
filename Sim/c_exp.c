#include <complex.h>

#ifdef WIN32
#define DECLSPEC __declspec(dllexport)
#else
#define DECLSPEC
#endif

typedef struct{
    double real;
    double imag;
}complex_data;

DECLSPEC double complex exp_wrapper(const double time, const double a, const double b){
    return cexp(I*time*(a - b));
    // complex_data res = { .real = creal(result), .imag = cimag(result)};
    // return res;
}