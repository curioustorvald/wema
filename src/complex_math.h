/*
 * WEMA - Complex number utilities
 * Header-only implementation with inline functions
 */

#ifndef COMPLEX_MATH_H
#define COMPLEX_MATH_H

#include "wema.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Create complex number */
static inline Complex cmplx(float re, float im) {
    return (Complex){re, im};
}

/* Complex addition */
static inline Complex cmplx_add(Complex a, Complex b) {
    return cmplx(a.re + b.re, a.im + b.im);
}

/* Complex subtraction */
static inline Complex cmplx_sub(Complex a, Complex b) {
    return cmplx(a.re - b.re, a.im - b.im);
}

/* Complex multiplication */
static inline Complex cmplx_mul(Complex a, Complex b) {
    return cmplx(a.re * b.re - a.im * b.im,
                 a.re * b.im + a.im * b.re);
}

/* Scale by real */
static inline Complex cmplx_scale(Complex a, float s) {
    return cmplx(a.re * s, a.im * s);
}

/* Complex magnitude */
static inline float cmplx_abs(Complex a) {
    return sqrtf(a.re * a.re + a.im * a.im);
}

/* Complex phase (angle) */
static inline float cmplx_phase(Complex a) {
    return atan2f(a.im, a.re);
}

/* Create from polar form */
static inline Complex cmplx_from_polar(float mag, float phase) {
    return cmplx(mag * cosf(phase), mag * sinf(phase));
}

/* Complex conjugate */
static inline Complex cmplx_conj(Complex a) {
    return cmplx(a.re, -a.im);
}

/* Phase unwrapping: wrap angle to [-pi, pi] */
static inline float phase_wrap(float phase) {
    while (phase > (float)M_PI)  phase -= 2.0f * (float)M_PI;
    while (phase < -(float)M_PI) phase += 2.0f * (float)M_PI;
    return phase;
}

/* Phase difference with unwrapping */
static inline float phase_diff(float prev, float curr) {
    return phase_wrap(curr - prev);
}

#endif /* COMPLEX_MATH_H */
