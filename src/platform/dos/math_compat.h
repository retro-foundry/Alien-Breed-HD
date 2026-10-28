#ifndef AB3D_DOS_MATH_COMPAT_H
#define AB3D_DOS_MATH_COMPAT_H
#include <math.h>
/* DJGPP's math library lacks these C99 entry points. Geometry inputs are
 * finite; retain the C99 NaN contract for other callers as well. */
static inline double ab3d_dos_fmin(double a,double b) {
    if(__builtin_isnan(a))return b;
    if(__builtin_isnan(b))return a;
    if(a==0 && b==0)return __builtin_signbit(a)?a:b;
    return a<b?a:b;
}
static inline double ab3d_dos_fmax(double a,double b) {
    if(__builtin_isnan(a))return b;
    if(__builtin_isnan(b))return a;
    if(a==0 && b==0)return __builtin_signbit(a)?b:a;
    return a>b?a:b;
}
#define fmin ab3d_dos_fmin
#define fmax ab3d_dos_fmax
#endif
