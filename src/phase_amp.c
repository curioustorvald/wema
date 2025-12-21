/*
 * WEMA - Phase Amplification
 *
 * Modifies wavelet coefficient phases to amplify motion.
 * Includes spatial smoothing of temporal deltas to reduce noise.
 */

#include "phase_amp.h"
#include "complex_math.h"
#include "alloc.h"

#include <math.h>
#include <string.h>

/*============================================================================
 * Spatial smoothing of delta coefficients (3x3 box filter per subband)
 *===========================================================================*/

static void smooth_subband_delta(const float *delta_in, float *delta_out,
                                  int width, int height) {
    /* 3x3 box filter - averages neighboring coefficient deltas
     * Real motion affects neighboring coefficients similarly
     * Noise is random and averages out */

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            int count = 0;

            /* 3x3 neighborhood */
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                        sum += delta_in[ny * width + nx];
                        count++;
                    }
                }
            }

            delta_out[y * width + x] = sum / (float)count;
        }
    }
}

void phase_amplify(const float *delta_coeff,
                   const float *orig_coeff,
                   float alpha,
                   DTCWTCoeffs *coeffs,
                   bool spatial_smooth) {
    /* Compute total positions for statistics and buffer allocation */
    size_t total_pos = 0;
    size_t max_subband_size = 0;

    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        size_t sub_size = (size_t)coeffs->subbands[lev][0].width *
                          coeffs->subbands[lev][0].height;
        total_pos += sub_size * WEMA_NUM_ORIENTATIONS;
        if (sub_size > max_subband_size) {
            max_subband_size = sub_size;
        }
    }

    /* Allocate smoothed delta buffer (only if smoothing enabled) */
    float *smoothed_delta = NULL;
    float *temp_in = NULL;
    float *temp_out = NULL;

    if (spatial_smooth) {
        smoothed_delta = mem_alloc(total_pos * sizeof(float));
        temp_in = mem_alloc(max_subband_size * sizeof(float));
        temp_out = mem_alloc(max_subband_size * sizeof(float));

        if (!smoothed_delta || !temp_in || !temp_out) {
            /* Fallback: use unsmoothed delta */
            mem_free(smoothed_delta);
            mem_free(temp_in);
            mem_free(temp_out);
            smoothed_delta = NULL;
            temp_in = NULL;
            temp_out = NULL;
        }
    }

    /* Apply spatial smoothing to each subband's delta coefficients */
    if (spatial_smooth && smoothed_delta) {
        size_t pos = 0;

        for (int lev = 0; lev < coeffs->num_levels; lev++) {
            int w = coeffs->subbands[lev][0].width;
            int h = coeffs->subbands[lev][0].height;
            int n = w * h;

            for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
                /* Copy this subband's deltas to temp buffer */
                memcpy(temp_in, delta_coeff + pos, n * sizeof(float));

                /* Apply 3x3 spatial smoothing */
                smooth_subband_delta(temp_in, temp_out, w, h);

                /* Store smoothed result */
                memcpy(smoothed_delta + pos, temp_out, n * sizeof(float));

                pos += n;
            }
        }
    }

    const float *delta_to_use = smoothed_delta ? smoothed_delta : delta_coeff;

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
                    new_val = orig + level_alpha * delta_to_use[pos];
                } else {
                    new_val = orig;
                }

                sub->coeffs[i] = cmplx(new_val, 0.0f);
                pos++;
            }
        }
    }

    mem_free(smoothed_delta);
    mem_free(temp_in);
    mem_free(temp_out);
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
                    new_phase = orig_phase + alpha * delta_phi[pos];
                } else {
                    new_phase = orig_phase;
                }

                sub->coeffs[i] = cmplx_from_polar(amp, new_phase);

                pos++;
            }
        }
    }
}
