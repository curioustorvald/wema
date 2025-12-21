/*
 * WEMA - Coefficient Amplification
 *
 * Modifies wavelet coefficients to amplify motion.
 */

#include "phase_amp.h"

#include <math.h>

void coeff_amplify(const float *delta_coeff,
                   const float *orig_coeff,
                   float alpha,
                   DWTCoeffs *coeffs) {
    /* Compute total positions for statistics */
    size_t total_pos = 0;
    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        size_t sub_size = (size_t)coeffs->subbands[lev][0].width *
                          coeffs->subbands[lev][0].height;
        total_pos += sub_size * WEMA_NUM_ORIENTATIONS;
    }

    /* Compute coefficient magnitude statistics for adaptive thresholding */
    float mag_sum = 0.0f;
    for (size_t i = 0; i < total_pos; i++) {
        float m = orig_coeff[i];
        mag_sum += (m > 0) ? m : -m;
    }
    float mag_mean = mag_sum / (float)total_pos;
    float coeff_threshold = mag_mean * 0.2f;  /* 20% of mean magnitude */

    /* Apply amplification */
    size_t pos = 0;

    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        /* Scale weighting - slight attenuation at finest level */
        float scale_weight = (lev == 0) ? 0.5f : 1.0f;
        float level_alpha = alpha * scale_weight;

        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            Subband *sub = &coeffs->subbands[lev][o];
            int n = sub->width * sub->height;

            for (int i = 0; i < n; i++) {
                float orig = orig_coeff[pos];
                float orig_abs = (orig > 0) ? orig : -orig;

                float new_val;
                if (level_alpha > 0.0f && orig_abs > coeff_threshold) {
                    new_val = orig + level_alpha * delta_coeff[pos];
                } else {
                    new_val = orig;
                }

                sub->coeffs[i] = new_val;
                pos++;
            }
        }
    }
}
