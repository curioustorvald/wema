/*
 * WEMA - Temporal Wavelet Filter
 *
 * Uses Haar wavelet for temporal decomposition.
 * Band-pass by zeroing wavelet scales outside frequency band.
 */

#include "temporal_filter.h"
#include "complex_math.h"
#include "alloc.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/*============================================================================
 * Haar wavelet 1D transform
 *===========================================================================*/

/* In-place Haar DWT */
static void haar_dwt_1d(float *data, int n) {
    float *temp = mem_alloc(n * sizeof(float));
    if (!temp) return;

    int len = n;
    while (len > 1) {
        int half = len / 2;
        for (int i = 0; i < half; i++) {
            float a = data[2 * i];
            float b = data[2 * i + 1];
            temp[i] = (a + b) * 0.5f;         /* Lowpass (average) */
            temp[half + i] = (a - b) * 0.5f;  /* Highpass (difference) */
        }
        memcpy(data, temp, len * sizeof(float));
        len = half;
    }

    mem_free(temp);
}

/* In-place Haar inverse DWT */
static void haar_idwt_1d(float *data, int n) {
    float *temp = mem_alloc(n * sizeof(float));
    if (!temp) return;

    int len = 2;
    while (len <= n) {
        int half = len / 2;
        for (int i = 0; i < half; i++) {
            float lo = data[i];
            float hi = data[half + i];
            temp[2 * i] = lo + hi;
            temp[2 * i + 1] = lo - hi;
        }
        memcpy(data, temp, len * sizeof(float));
        len *= 2;
    }

    mem_free(temp);
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
                         const DTCWTCoeffs *coeffs,
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
                          const DTCWTCoeffs *coeffs) {
    size_t pos = 0;

    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            const Subband *sub = &coeffs->subbands[lev][o];
            int n = sub->width * sub->height;

            for (int i = 0; i < n; i++) {
                Complex c = sub->coeffs[i];

                /* Store the real coefficient value directly (not phase).
                 * For Haar wavelets, phase-based amplification doesn't work
                 * because we don't have true complex wavelets. Instead we
                 * band-pass filter and amplify the coefficient values. */
                size_t buf_idx = pos * buf->window_size + buf->head;
                buf->phase_buffer[buf_idx] = c.re;

                /* Store current value as "amplitude" for reconstruction */
                buf->amplitude[pos] = c.re;

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
                           float *delta_phi_filtered) {
    if (buf->filled < buf->window_size) {
        /* Not enough data yet */
        memset(delta_phi_filtered, 0, buf->num_positions * sizeof(float));
        return;
    }

    int ws = buf->window_size;
    int center = ws / 2;

    /* Allocate work buffer for temporal DWT */
    float *temp = mem_alloc(ws * sizeof(float));
    if (!temp) {
        memset(delta_phi_filtered, 0, buf->num_positions * sizeof(float));
        return;
    }

    /*
     * Compute frequency range for each wavelet scale.
     * Scale j corresponds to frequencies around fs / 2^(j+1)
     * where fs = 1.0 (normalized sample rate).
     *
     * level 0: 0.25 - 0.5  (highest freq)
     * level 1: 0.125 - 0.25
     * level 2: 0.0625 - 0.125
     * etc.
     */
    int num_levels = filt->temporal_levels;
    bool *keep_level = mem_alloc(num_levels * sizeof(bool));
    if (!keep_level) {
        mem_free(temp);
        memset(delta_phi_filtered, 0, buf->num_positions * sizeof(float));
        return;
    }

    for (int j = 0; j < num_levels; j++) {
        float f_high_scale = 0.5f / (float)(1 << j);
        float f_low_scale = 0.5f / (float)(1 << (j + 1));

        /* Keep this level if it overlaps with desired band */
        keep_level[j] = (f_high_scale >= filt->f_low_norm &&
                         f_low_scale <= filt->f_high_norm);
    }

    for (size_t pos = 0; pos < buf->num_positions; pos++) {
        /* Extract coefficient history from circular buffer */
        size_t base = pos * ws;
        int start = (buf->head + ws) % ws;  /* Oldest sample */

        for (int t = 0; t < ws; t++) {
            int idx = (start + t) % ws;
            temp[t] = buf->phase_buffer[base + idx];
        }

        /* Remove DC (mean value) - we only want the temporal variations */
        float mean = 0.0f;
        for (int t = 0; t < ws; t++) {
            mean += temp[t];
        }
        mean /= (float)ws;

        for (int t = 0; t < ws; t++) {
            temp[t] -= mean;
        }

        /* Apply temporal DWT */
        haar_dwt_1d(temp, ws);

        /* Zero out scales outside frequency band */
        /* temp[0] is the DC coefficient (lowest freq) */
        temp[0] = 0.0f;  /* Always remove DC */

        /* Detail coefficients at each level */
        int level_start = 1;
        for (int j = num_levels - 1; j >= 0; j--) {
            int level_size = 1 << j;  /* Number of coefficients at this level */
            if (!keep_level[j]) {
                for (int k = 0; k < level_size; k++) {
                    temp[level_start + k] = 0.0f;
                }
            }
            level_start += level_size;
        }

        /* Inverse DWT */
        haar_idwt_1d(temp, ws);

        /* Extract center sample (current frame with half-window latency) */
        delta_phi_filtered[pos] = temp[center];
    }

    mem_free(keep_level);
    mem_free(temp);
}

int temporal_buffer_output_count(const TemporalBuffer *buf) {
    if (buf->filled < buf->window_size) {
        return 0;
    }
    return buf->filled - buf->window_size / 2;
}
