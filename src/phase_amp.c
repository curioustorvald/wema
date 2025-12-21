/*
 * WEMA - Phase Amplification
 *
 * Modifies wavelet coefficient phases to amplify motion.
 */

#include "phase_amp.h"
#include "complex_math.h"

#include <math.h>

void phase_amplify(const float *delta_phi,
                   const float *amplitude,
                   float alpha,
                   DTCWTCoeffs *coeffs) {
    size_t pos = 0;

    /* Compute amplitude statistics for adaptive thresholding */
    size_t total_pos = 0;
    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        total_pos += (size_t)coeffs->subbands[lev][0].width *
                     coeffs->subbands[lev][0].height * WEMA_NUM_ORIENTATIONS;
    }

    /* Find median-ish amplitude for threshold (use mean as approximation) */
    float amp_sum = 0.0f;
    for (size_t i = 0; i < total_pos; i++) {
        amp_sum += amplitude[i];
    }
    float amp_mean = amp_sum / (float)total_pos;
    float amp_threshold = amp_mean * 0.01f;  /* 30% of mean amplitude - more aggressive */

    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        /* Spatial scale weighting:
         * Fine scales capture texture/noise, coarse scales capture motion.
         * Only amplify at coarser scales (level 2+) */
        float scale_weight;
        if (lev <= 1) {
            scale_weight = 0.0f;   /* Skip two finest levels */
        } else if (lev == 2) {
            scale_weight = 0.5f;   /* Attenuate level 2 */
        } else {
            scale_weight = 1.0f;   /* Full amplification for coarsest scales */
        }

        float level_alpha = alpha * scale_weight;

        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            Subband *sub = &coeffs->subbands[lev][o];
            int n = sub->width * sub->height;

            for (int i = 0; i < n; i++) {
                Complex c = sub->coeffs[i];
                float orig_phase = cmplx_phase(c);
                float amp = amplitude[pos];

                float new_phase;
                if (amp > amp_threshold && level_alpha > 0.0f) {
                    /* Amplify phase for significant coefficients */
                    new_phase = orig_phase + level_alpha * delta_phi[pos];
                } else {
                    /* Below threshold or attenuated level: no change */
                    new_phase = orig_phase;
                }

                /* Reconstruct complex coefficient */
                sub->coeffs[i] = cmplx_from_polar(amp, new_phase);

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
