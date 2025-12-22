/*
 * WEMA - Coefficient Amplification
 *
 * Modifies wavelet coefficients to amplify motion.
 * Optimized for autovectorization.
 */

#include "phase_amp.h"

#include <math.h>

void coeff_amplify(const float * restrict delta_coeff,
                   const float * restrict orig_coeff,
                   float alpha,
                   DWTCoeffs *coeffs) {
    /* Compute total positions for statistics */
    size_t total_pos = 0;
    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        size_t sub_size = (size_t)coeffs->subbands[lev][0].width *
                          coeffs->subbands[lev][0].height;
        total_pos += sub_size * WEMA_NUM_ORIENTATIONS;
    }

    /* Compute coefficient magnitude statistics - vectorizable */
    float mag_sum = 0.0f;
    for (size_t i = 0; i < total_pos; i++) {
        mag_sum += fabsf(orig_coeff[i]);
    }
    const float coeff_threshold = mag_sum / (float)total_pos * 0.2f;

    /* Apply amplification per level */
    size_t pos = 0;

    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        /* Scale weighting - slight attenuation at finest level */
        const float level_alpha = alpha * ((lev == 0) ? 0.5f : 1.0f);

        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            Subband *sub = &coeffs->subbands[lev][o];
            float * restrict out = sub->coeffs;
            const int n = sub->width * sub->height;

            /* Vectorizable inner loop using branchless select */
            for (int i = 0; i < n; i++) {
                const float orig = orig_coeff[pos];
                const float delta = delta_coeff[pos];

                /* Branchless: apply amplification only if above threshold */
                const float mask = (fabsf(orig) > coeff_threshold) ? 1.0f : 0.0f;
                out[i] = orig + mask * level_alpha * delta;

                pos++;
            }
        }
    }
}
