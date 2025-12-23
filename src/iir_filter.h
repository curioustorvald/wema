/*
 * IIR Band-pass Filter for Temporal Filtering
 *
 * Replaces Haar DWT-based temporal filtering with a direct IIR approach.
 * Uses cascaded 2nd-order Butterworth sections (biquads).
 */

#ifndef IIR_FILTER_H
#define IIR_FILTER_H

#include <stddef.h>
#include <stdbool.h>

/* Maximum filter order (number of biquad sections) */
#define IIR_MAX_SECTIONS 4

/*
 * Biquad section coefficients (Direct Form II Transposed)
 * H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
 */
typedef struct {
    float b0, b1, b2;
    float a1, a2;
} BiquadCoeffs;

/*
 * Per-position filter state (two delays per biquad section)
 */
typedef struct {
    float w1, w2;  /* Delay elements */
} BiquadState;

/*
 * IIR Filter configuration
 */
typedef struct {
    int num_sections;                        /* Number of cascaded biquads */
    BiquadCoeffs sections[IIR_MAX_SECTIONS]; /* Filter coefficients */
} IIRFilterCoeffs;

/*
 * IIR Temporal Filter context
 */
typedef struct {
    IIRFilterCoeffs coeffs;     /* Filter coefficients (shared) */
    size_t num_positions;       /* Number of coefficient positions */
    BiquadState *state;         /* Per-position state [num_positions * num_sections] */
    int warmup_frames;          /* Frames needed for filter to stabilize */
    int frames_processed;       /* Frames processed so far */
} IIRTemporalFilter;

/*
 * Design a Butterworth band-pass filter.
 *
 * @param coeffs    Output filter coefficients
 * @param f_low     Low cutoff frequency (Hz)
 * @param f_high    High cutoff frequency (Hz)
 * @param fs        Sampling frequency (Hz)
 * @param order     Filter order (2 or 4 recommended)
 * @return 0 on success, -1 on error
 */
int iir_design_bandpass(IIRFilterCoeffs *coeffs,
                        float f_low, float f_high, float fs, int order);

/*
 * Initialize IIR temporal filter.
 *
 * @param filt          Filter context
 * @param num_positions Number of coefficient positions to filter
 * @param f_low         Low cutoff frequency (Hz)
 * @param f_high        High cutoff frequency (Hz)
 * @param fs            Sampling frequency (fps)
 * @return 0 on success, -1 on error
 */
int iir_temporal_init(IIRTemporalFilter *filt, size_t num_positions,
                      float f_low, float f_high, float fs);

/*
 * Free IIR temporal filter resources.
 */
void iir_temporal_free(IIRTemporalFilter *filt);

/*
 * Process one frame of coefficients through the IIR filter.
 *
 * @param filt          Filter context
 * @param input         Input coefficients [num_positions]
 * @param output        Output filtered delta [num_positions]
 */
void iir_temporal_process(IIRTemporalFilter *filt,
                          const float *input, float *output);

/*
 * Check if filter has warmed up (output is valid).
 */
bool iir_temporal_ready(const IIRTemporalFilter *filt);

#endif /* IIR_FILTER_H */
