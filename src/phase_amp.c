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

    /* Compute coefficient magnitude statistics for adaptive thresholding */
    size_t total_pos = 0;
    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        total_pos += (size_t)coeffs->subbands[lev][0].width *
                     coeffs->subbands[lev][0].height * WEMA_NUM_ORIENTATIONS;
    }

    float mag_sum = 0.0f;
    for (size_t i = 0; i < total_pos; i++) {
        float m = orig_coeff[i];
        mag_sum += (m > 0) ? m : -m;  /* Absolute value */
    }
    float mag_mean = mag_sum / (float)total_pos;
    float coeff_threshold = mag_mean * 0.2f;  /* 20% of mean magnitude */

    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        /* Minimal spatial scale weighting:
         * CDF 9/7 is smooth enough that we only need slight attenuation
         * at the finest level to reduce texture amplification */
        float scale_weight;
        if (lev == 0) {
            scale_weight = 0.5f;   /* Slight attenuation at finest level */
        } else {
            scale_weight = 1.0f;   /* Full amplification for all other levels */
        }

        float level_alpha = alpha * scale_weight;

        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            Subband *sub = &coeffs->subbands[lev][o];
            int n = sub->width * sub->height;

            for (int i = 0; i < n; i++) {
                float orig = orig_coeff[pos];
                float orig_abs = (orig > 0) ? orig : -orig;

                float new_val;
                if (level_alpha > 0.0f && orig_abs > coeff_threshold) {
                    /* Amplify significant coefficients at non-fine scales */
                    new_val = orig + level_alpha * delta_coeff[pos];
                } else {
                    /* Keep original for fine scales or weak coefficients */
                    new_val = orig;
                }

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
