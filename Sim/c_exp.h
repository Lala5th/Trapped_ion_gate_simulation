#ifndef C_EXP_H
#define C_EXP_H

#include <complex.h>

#ifdef WIN32
#define DECLSPEC __declspec(dllimport)
#else
#define DECLSPEC
#endif

DECLSPEC double complex exp_wrapper(double,double,double);

#endif