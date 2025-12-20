/*
 * WEMA - Temporal Wavelet Filter
 *
 * Implements temporal band-pass filtering using 1D wavelets.
 */

#ifndef TEMPORAL_FILTER_H
#define TEMPORAL_FILTER_H

#include "wema.h"

/*
 * Initialize temporal filter.
 * f_low, f_high: cutoff frequencies (normalized: 0 to 0.5)
 * window_size: temporal buffer size (power of 2)
 * Returns 0 on success, -1 on error.
 */
int temporal_filter_init(TemporalFilter *filt,
                         float f_low, float f_high,
                         int window_size);

/*
 * Free temporal filter resources.
 */
void temporal_filter_free(TemporalFilter *filt);

/*
 * Initialize temporal buffer for phase storage.
 */
int temporal_buffer_init(TemporalBuffer *buf,
                         const DTCWTCoeffs *coeffs,
                         int window_size);

/*
 * Free temporal buffer resources.
 */
void temporal_buffer_free(TemporalBuffer *buf);

/*
 * Push new frame's phase values into temporal buffer.
 */
void temporal_buffer_push(TemporalBuffer *buf,
                          const DTCWTCoeffs *coeffs);

/*
 * Apply temporal band-pass filter.
 * Returns filtered phase delta for amplification.
 * output: array [num_positions]
 */
void temporal_filter_apply(const TemporalFilter *filt,
                           const TemporalBuffer *buf,
                           float *delta_phi_filtered);

/*
 * Check if buffer has enough frames for filtering.
 */
static inline bool temporal_buffer_ready(const TemporalBuffer *buf) {
    return buf->filled >= buf->window_size;
}

/*
 * Get number of output frames available (accounting for latency).
 */
int temporal_buffer_output_count(const TemporalBuffer *buf);

#endif /* TEMPORAL_FILTER_H */
