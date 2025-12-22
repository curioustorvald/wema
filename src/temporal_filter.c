/*
 * WEMA - Temporal Wavelet Filter
 *
 * Uses Haar wavelet for temporal decomposition.
 * Band-pass by zeroing wavelet scales outside frequency band.
 */

#include "temporal_filter.h"
#include "alloc.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/*============================================================================
 * Haar wavelet 1D transform (optimized - no allocation)
 *===========================================================================*/

/* In-place Haar DWT using external temp buffer */
static void haar_dwt_1d(float * restrict data, int n, float * restrict temp) {
    int len = n;
    while (len > 1) {
        const int half = len / 2;
        for (int i = 0; i < half; i++) {
            const float a = data[2 * i];
            const float b = data[2 * i + 1];
            temp[i] = (a + b) * 0.5f;
            temp[half + i] = (a - b) * 0.5f;
        }
        memcpy(data, temp, len * sizeof(float));
        len = half;
    }
}

/* In-place Haar inverse DWT using external temp buffer */
static void haar_idwt_1d(float * restrict data, int n, float * restrict temp) {
    int len = 2;
    while (len <= n) {
        const int half = len / 2;
        for (int i = 0; i < half; i++) {
            const float lo = data[i];
            const float hi = data[half + i];
            temp[2 * i] = lo + hi;
            temp[2 * i + 1] = lo - hi;
        }
        memcpy(data, temp, len * sizeof(float));
        len *= 2;
    }
}

/*============================================================================
 * Temporal filter
 *===========================================================================*/

int temporal_filter_init(TemporalFilter *filt,
                         float f_low, float f_high,
                         int window_size) {
    memset(filt, 0, sizeof(*filt));

    filt->f_low_norm = f_low;
    filt->f_high_norm = f_high;

    /* Compute number of temporal levels */
    int levels = 0;
    int ws = window_size;
    while (ws > 1) {
        ws /= 2;
        levels++;
    }
    filt->temporal_levels = levels;

    /* Haar wavelet: simple coefficients */
    filt->filter_len = 2;
    filt->lp_filter = mem_alloc(2 * sizeof(float));
    filt->hp_filter = mem_alloc(2 * sizeof(float));

    if (!filt->lp_filter || !filt->hp_filter) {
        temporal_filter_free(filt);
        return -1;
    }

    filt->lp_filter[0] = 0.5f;
    filt->lp_filter[1] = 0.5f;
    filt->hp_filter[0] = 0.5f;
    filt->hp_filter[1] = -0.5f;

    return 0;
}

void temporal_filter_free(TemporalFilter *filt) {
    mem_free(filt->lp_filter);
    mem_free(filt->hp_filter);
    filt->lp_filter = NULL;
    filt->hp_filter = NULL;
}

int temporal_buffer_init(TemporalBuffer *buf,
                         const DWTCoeffs *coeffs,
                         int window_size) {
    memset(buf, 0, sizeof(*buf));

    buf->window_size = window_size;
    buf->num_levels = coeffs->num_levels;

    /* Count total positions */
    size_t total = 0;
    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        buf->widths[lev] = coeffs->subbands[lev][0].width;
        buf->heights[lev] = coeffs->subbands[lev][0].height;
        total += (size_t)buf->widths[lev] * buf->heights[lev] * WEMA_NUM_ORIENTATIONS;
    }
    buf->num_positions = total;

    /* Allocate circular phase buffer */
    buf->phase_buffer = mem_calloc(total * window_size, sizeof(float));
    buf->amplitude = mem_calloc(total, sizeof(float));
    buf->dc_phase = mem_calloc(total, sizeof(float));

    if (!buf->phase_buffer || !buf->amplitude || !buf->dc_phase) {
        temporal_buffer_free(buf);
        return -1;
    }

    buf->head = 0;
    buf->filled = 0;

    return 0;
}

void temporal_buffer_free(TemporalBuffer *buf) {
    mem_free(buf->phase_buffer);
    mem_free(buf->amplitude);
    mem_free(buf->dc_phase);
    buf->phase_buffer = NULL;
    buf->amplitude = NULL;
    buf->dc_phase = NULL;
}

void temporal_buffer_push(TemporalBuffer *buf,
                          const DWTCoeffs *coeffs) {
    size_t pos = 0;

    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            const Subband *sub = &coeffs->subbands[lev][o];
            int n = sub->width * sub->height;

            for (int i = 0; i < n; i++) {
                float c = sub->coeffs[i];

                /* Store coefficient value for temporal filtering */
                size_t buf_idx = pos * buf->window_size + buf->head;
                buf->phase_buffer[buf_idx] = c;

                /* Store current value for reconstruction */
                buf->amplitude[pos] = c;

                pos++;
            }
        }
    }

    /* Advance head */
    buf->head = (buf->head + 1) % buf->window_size;
    if (buf->filled < buf->window_size) {
        buf->filled++;
    }
}

void temporal_filter_apply(const TemporalFilter *filt,
                           const TemporalBuffer *buf,
                           float * restrict delta_phi_filtered,
                           bool bilateral) {
    if (buf->filled < buf->window_size) {
        memset(delta_phi_filtered, 0, buf->num_positions * sizeof(float));
        return;
    }

    const int ws = buf->window_size;
    const int center = ws / 2;
    const int num_levels = filt->temporal_levels;
    const float inv_ws = 1.0f / (float)ws;
    const float * restrict phase_buf = buf->phase_buffer;

    /* Pre-allocate all work buffers once */
    float * restrict temp = mem_alloc(ws * sizeof(float));
    float * restrict haar_work = mem_alloc(ws * sizeof(float));
    float * restrict weights = bilateral ? mem_alloc(ws * sizeof(float)) : NULL;
    float * restrict orig_vals = bilateral ? mem_alloc(ws * sizeof(float)) : NULL;

    if (!temp || !haar_work || (bilateral && (!weights || !orig_vals))) {
        mem_free(temp);
        mem_free(haar_work);
        mem_free(weights);
        mem_free(orig_vals);
        memset(delta_phi_filtered, 0, buf->num_positions * sizeof(float));
        return;
    }

    /* Precompute which levels to keep */
    bool keep_level[16];  /* Max 16 levels (2^16 = 64K window) */
    for (int j = 0; j < num_levels && j < 16; j++) {
        float f_high_scale = 0.5f / (float)(1 << j);
        float f_low_scale = 0.5f / (float)(1 << (j + 1));
        keep_level[j] = (f_high_scale >= filt->f_low_norm &&
                         f_low_scale <= filt->f_high_norm);
    }

    /* Precompute time weights for bilateral (Gaussian, position-independent) */
    float time_weights[256];  /* Max window size */
    if (bilateral) {
        const float time_sigma = (float)ws * 0.15f;
        const float inv_2sigma_sq = 1.0f / (2.0f * time_sigma * time_sigma);
        for (int t = 0; t < ws; t++) {
            float time_diff = (float)(t - center);
            time_weights[t] = expf(-(time_diff * time_diff) * inv_2sigma_sq);
        }
    }

    /* Precompute circular buffer indices */
    const int start = (buf->head + ws) % ws;

    /* Main processing loop */
    for (size_t pos = 0; pos < buf->num_positions; pos++) {
        const size_t base = pos * ws;

        /* Extract coefficient history - unroll for common window sizes */
        float mean = 0.0f;
        if (start == 0) {
            /* Contiguous case - vectorizable */
            for (int t = 0; t < ws; t++) {
                float val = phase_buf[base + t];
                temp[t] = val;
                mean += val;
            }
        } else {
            /* Wrapped case */
            for (int t = 0; t < ws; t++) {
                int idx = (start + t) % ws;
                float val = phase_buf[base + idx];
                temp[t] = val;
                mean += val;
            }
        }

        const float current_val = temp[center];
        mean *= inv_ws;

        /* DC removal - vectorizable */
        for (int t = 0; t < ws; t++) {
            temp[t] -= mean;
        }

        /* Apply temporal DWT */
        haar_dwt_1d(temp, ws, haar_work);

        /* Zero out DC and unwanted scales */
        temp[0] = 0.0f;
        int level_start = 1;
        for (int j = num_levels - 1; j >= 0; j--) {
            int level_size = 1 << j;
            if (!keep_level[j]) {
                memset(temp + level_start, 0, level_size * sizeof(float));
            }
            level_start += level_size;
        }

        /* Inverse DWT */
        haar_idwt_1d(temp, ws, haar_work);

        if (bilateral) {
            /* Re-extract original for bilateral weighting */
            if (start == 0) {
                memcpy(orig_vals, phase_buf + base, ws * sizeof(float));
            } else {
                for (int t = 0; t < ws; t++) {
                    orig_vals[t] = phase_buf[base + (start + t) % ws];
                }
            }

            /* Compute MAD for adaptive sigma */
            float mad_sum = 0.0f;
            for (int t = 0; t < ws; t++) {
                float dev = orig_vals[t] - current_val;
                mad_sum += (dev > 0.0f) ? dev : -dev;
            }
            float sigma = fmaxf(mad_sum * inv_ws * 0.5f, 0.001f);
            float inv_2sigma_sq = 1.0f / (2.0f * sigma * sigma);

            /* Compute bilateral weights */
            float weight_sum = 0.0f;
            for (int t = 0; t < ws; t++) {
                float diff = orig_vals[t] - current_val;
                float range_weight = expf(-(diff * diff) * inv_2sigma_sq);
                weights[t] = range_weight * time_weights[t];
                weight_sum += weights[t];
            }

            /* Weighted average */
            if (weight_sum > 1e-8f) {
                float inv_weight_sum = 1.0f / weight_sum;
                float result = 0.0f;
                for (int t = 0; t < ws; t++) {
                    result += weights[t] * temp[t];
                }
                delta_phi_filtered[pos] = result * inv_weight_sum;
            } else {
                delta_phi_filtered[pos] = temp[center];
            }
        } else {
            delta_phi_filtered[pos] = temp[center];
        }
    }

    mem_free(orig_vals);
    mem_free(weights);
    mem_free(haar_work);
    mem_free(temp);
}

int temporal_buffer_output_count(const TemporalBuffer *buf) {
    if (buf->filled < buf->window_size) {
        return 0;
    }
    return buf->filled - buf->window_size / 2;
}
