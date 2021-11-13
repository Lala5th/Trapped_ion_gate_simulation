#include <complex>

#ifdef WIN32
#define DECLSPEC __declspec(dllexport)
#else
#define DECLSPEC
#endif

struct complex {
    double real;
    double imag;
};

extern "C" DECLSPEC complex exp_wrapper(double time, double a, double b){
    using namespace std::complex_literals;
    std::complex<double> result = std::exp(1.i*((std::complex<double>)time*(a - b)));
    return complex{result.real(), result.imag()};
}