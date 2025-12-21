/*
 * WEMA - Phase Amplification
 *
 * Modifies wavelet coefficient phases to amplify motion.
 */

#include "phase_amp.h"
#include "complex_math.h"

#include <math.h>

void phase_amplify(const float *delta_coeff,
                   const float *orig_coeff,
                   float alpha,
                   DTCWTCoeffs *coeffs) {
    size_t pos = 0;

    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            Subband *sub = &coeffs->subbands[lev][o];
            int n = sub->width * sub->height;

            for (int i = 0; i < n; i++) {
                /* Direct coefficient amplification:
                 * new_coeff = orig_coeff + alpha * filtered_delta
                 * This amplifies temporal variations in the wavelet coefficients
                 * which correspond to motion in the spatial domain. */
                float new_val = orig_coeff[pos] + alpha * delta_coeff[pos];

                /* Store back (imaginary part doesn't matter for reconstruction) */
                sub->coeffs[i] = cmplx(new_val, 0.0f);

                pos++;
            }
        }
    }
}

void phase_amplify_threshold(const float *delta_phi,
                             const float *amplitude,
                             float alpha,
                             float threshold,
                             DTCWTCoeffs *coeffs) {
    size_t pos = 0;

    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            Subband *sub = &coeffs->subbands[lev][o];
            int n = sub->width * sub->height;

            for (int i = 0; i < n; i++) {
                Complex c = sub->coeffs[i];
                float orig_phase = cmplx_phase(c);
                float amp = amplitude[pos];

                float new_phase;
                if (amp > threshold) {
                    /* Amplify */
                    new_phase = orig_phase + alpha * delta_phi[pos];
                } else {
                    /* Below threshold: no amplification */
                    new_phase = orig_phase;
                }

                sub->coeffs[i] = cmplx_from_polar(amp, new_phase);

                pos++;
            }
        }
    }
}
