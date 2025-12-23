/*
 * IIR Band-pass Filter Implementation
 */

#include "iir_filter.h"
#include "alloc.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Design a 2nd-order Butterworth low-pass biquad section.
 * Uses bilinear transform with frequency pre-warping.
 */
static void design_lowpass_biquad(BiquadCoeffs *c, float fc, float fs) {
    float wc = tanf((float)M_PI * fc / fs);  /* Pre-warped frequency */
    float wc2 = wc * wc;
    float sqrt2 = 1.41421356237f;
    float norm = 1.0f / (1.0f + sqrt2 * wc + wc2);

    c->b0 = wc2 * norm;
    c->b1 = 2.0f * c->b0;
    c->b2 = c->b0;
    c->a1 = 2.0f * (wc2 - 1.0f) * norm;
    c->a2 = (1.0f - sqrt2 * wc + wc2) * norm;
}

/*
 * Design a 2nd-order Butterworth high-pass biquad section.
 */
static void design_highpass_biquad(BiquadCoeffs *c, float fc, float fs) {
    float wc = tanf((float)M_PI * fc / fs);
    float wc2 = wc * wc;
    float sqrt2 = 1.41421356237f;
    float norm = 1.0f / (1.0f + sqrt2 * wc + wc2);

    c->b0 = norm;
    c->b1 = -2.0f * norm;
    c->b2 = norm;
    c->a1 = 2.0f * (wc2 - 1.0f) * norm;
    c->a2 = (1.0f - sqrt2 * wc + wc2) * norm;
}

/*
 * Design band-pass as cascade of high-pass and low-pass.
 * Order 2: 1 HP + 1 LP section (4th order overall)
 * Order 4: 2 HP + 2 LP sections (8th order overall)
 */
int iir_design_bandpass(IIRFilterCoeffs *coeffs,
                        float f_low, float f_high, float fs, int order) {
    memset(coeffs, 0, sizeof(*coeffs));

    if (f_low >= f_high || f_low <= 0 || f_high >= fs / 2) {
        return -1;
    }

    /* For simplicity, use order 2 (one HP + one LP biquad) */
    if (order < 2) order = 2;
    if (order > 4) order = 4;

    int sections_per_type = order / 2;
    coeffs->num_sections = sections_per_type * 2;

    if (coeffs->num_sections > IIR_MAX_SECTIONS) {
        coeffs->num_sections = IIR_MAX_SECTIONS;
        sections_per_type = IIR_MAX_SECTIONS / 2;
    }

    /* High-pass sections (remove frequencies below f_low) */
    for (int i = 0; i < sections_per_type; i++) {
        design_highpass_biquad(&coeffs->sections[i], f_low, fs);
    }

    /* Low-pass sections (remove frequencies above f_high) */
    for (int i = 0; i < sections_per_type; i++) {
        design_lowpass_biquad(&coeffs->sections[sections_per_type + i], f_high, fs);
    }

    return 0;
}

int iir_temporal_init(IIRTemporalFilter *filt, size_t num_positions,
                      float f_low, float f_high, float fs) {
    memset(filt, 0, sizeof(*filt));

    /* Design filter */
    if (iir_design_bandpass(&filt->coeffs, f_low, f_high, fs, 2) < 0) {
        return -1;
    }

    filt->num_positions = num_positions;

    /* Allocate state for all positions and sections */
    size_t state_count = num_positions * filt->coeffs.num_sections;
    filt->state = mem_calloc(state_count, sizeof(BiquadState));
    if (!filt->state) {
        return -1;
    }

    /* Warmup time: ~3 time constants of the lowest frequency */
    /* Time constant ≈ 1/(2*pi*f_low), need ~3 of these */
    filt->warmup_frames = (int)(3.0f * fs / (2.0f * (float)M_PI * f_low));
    if (filt->warmup_frames < 16) filt->warmup_frames = 16;

    return 0;
}

void iir_temporal_free(IIRTemporalFilter *filt) {
    mem_free(filt->state);
    filt->state = NULL;
    filt->num_positions = 0;
}

/*
 * Apply a single biquad section (Direct Form II Transposed)
 * This form has better numerical properties and is SIMD-friendly.
 */
static inline float biquad_process(const BiquadCoeffs *c, BiquadState *s, float x) {
    float y = c->b0 * x + s->w1;
    s->w1 = c->b1 * x - c->a1 * y + s->w2;
    s->w2 = c->b2 * x - c->a2 * y;
    return y;
}

void iir_temporal_process(IIRTemporalFilter *filt,
                          const float *input, float *output) {
    const int num_sections = filt->coeffs.num_sections;
    const BiquadCoeffs *coeffs = filt->coeffs.sections;
    BiquadState *state = filt->state;
    const size_t n = filt->num_positions;

    /* Process each position through the filter cascade */
    for (size_t i = 0; i < n; i++) {
        float x = input[i];

        /* Apply each biquad section in series */
        BiquadState *pos_state = &state[i * num_sections];
        for (int s = 0; s < num_sections; s++) {
            x = biquad_process(&coeffs[s], &pos_state[s], x);
        }

        output[i] = x;
    }

    filt->frames_processed++;
}

void iir_temporal_process_batch(IIRTemporalFilter *filt,
                                const float *input, float *output,
                                int batch_size) {
    const int num_sections = filt->coeffs.num_sections;
    const BiquadCoeffs *coeffs = filt->coeffs.sections;
    BiquadState *state = filt->state;
    const size_t n = filt->num_positions;

    /*
     * Batch processing strategy:
     * - Outer loop: parallel across positions (independent filter states)
     * - Inner loop: sequential across frames (maintains IIR state)
     *
     * Memory layout: input/output are [batch_size][num_positions] (frame-major)
     * Access pattern: For position p, frames are at offsets p, n+p, 2n+p, ...
     */
    #pragma omp parallel for schedule(static)
    for (size_t pos = 0; pos < n; pos++) {
        BiquadState *pos_state = &state[pos * num_sections];

        /* Process all frames for this position sequentially */
        for (int f = 0; f < batch_size; f++) {
            size_t idx = (size_t)f * n + pos;
            float x = input[idx];

            /* Apply each biquad section in series */
            for (int s = 0; s < num_sections; s++) {
                x = biquad_process(&coeffs[s], &pos_state[s], x);
            }

            output[idx] = x;
        }
    }

    filt->frames_processed += batch_size;
}

bool iir_temporal_ready(const IIRTemporalFilter *filt) {
    return filt->frames_processed >= filt->warmup_frames;
}
