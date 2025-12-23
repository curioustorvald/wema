/*
 * IIR Band-pass Filter Implementation
 * With AVX-512 SIMD optimization for batch processing
 */

#include "iir_filter.h"
#include "alloc.h"

#include <math.h>
#include <string.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* AVX-512 support detection and includes */
#if defined(__AVX512F__)
#define USE_AVX512 1
#include <immintrin.h>
#else
#define USE_AVX512 0
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* AVX-512 processes 16 floats at once */
#define SIMD_WIDTH 16

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
    const int num_sections = filt->coeffs.num_sections;

#if USE_AVX512
    /* Allocate SoA state for SIMD processing */
    filt->state_soa = mem_alloc(num_sections * sizeof(BiquadStateSoA));
    if (!filt->state_soa) {
        return -1;
    }

    /* Allocate aligned arrays for each section's w1 and w2 */
    for (int s = 0; s < num_sections; s++) {
        /* Align to 64 bytes for AVX-512 */
        filt->state_soa[s].w1 = mem_calloc(num_positions + SIMD_WIDTH, sizeof(float));
        filt->state_soa[s].w2 = mem_calloc(num_positions + SIMD_WIDTH, sizeof(float));
        if (!filt->state_soa[s].w1 || !filt->state_soa[s].w2) {
            iir_temporal_free(filt);
            return -1;
        }
    }
    filt->use_simd = true;
    filt->state = NULL;  /* Not used in SIMD mode */
#else
    /* Allocate AoS state for scalar processing */
    size_t state_count = num_positions * num_sections;
    filt->state = mem_calloc(state_count, sizeof(BiquadState));
    if (!filt->state) {
        return -1;
    }
    filt->state_soa = NULL;
    filt->use_simd = false;
#endif

    /*
     * Warmup time: time constants of the lowest frequency.
     * 3τ = 95% settled, 5τ = 99.3%, 7τ = 99.9%
     * With high amplification factors, we need more settling.
     */
    filt->warmup_frames = (int)(7.0f * fs / (2.0f * (float)M_PI * f_low));
    if (filt->warmup_frames < 30) filt->warmup_frames = 30;

    return 0;
}

void iir_temporal_free(IIRTemporalFilter *filt) {
    if (filt->state_soa) {
        for (int s = 0; s < filt->coeffs.num_sections; s++) {
            mem_free(filt->state_soa[s].w1);
            mem_free(filt->state_soa[s].w2);
        }
        mem_free(filt->state_soa);
        filt->state_soa = NULL;
    }
    mem_free(filt->state);
    filt->state = NULL;
    filt->num_positions = 0;
}

/*
 * Apply a single biquad section (Direct Form II Transposed)
 * Scalar version for fallback.
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
    const size_t n = filt->num_positions;

#if USE_AVX512
    if (filt->use_simd) {
        BiquadStateSoA *state_soa = filt->state_soa;
        const size_t n_simd = n & ~(size_t)(SIMD_WIDTH - 1);

        /* Process 16 positions at a time with AVX-512 */
        for (size_t i = 0; i < n_simd; i += SIMD_WIDTH) {
            __m512 x = _mm512_loadu_ps(&input[i]);

            /* Apply each biquad section in series */
            for (int s = 0; s < num_sections; s++) {
                const BiquadCoeffs *c = &coeffs[s];
                BiquadStateSoA *st = &state_soa[s];

                __m512 w1 = _mm512_loadu_ps(&st->w1[i]);
                __m512 w2 = _mm512_loadu_ps(&st->w2[i]);

                __m512 b0 = _mm512_set1_ps(c->b0);
                __m512 b1 = _mm512_set1_ps(c->b1);
                __m512 b2 = _mm512_set1_ps(c->b2);
                __m512 a1 = _mm512_set1_ps(c->a1);
                __m512 a2 = _mm512_set1_ps(c->a2);

                /* y = b0 * x + w1 */
                __m512 y = _mm512_fmadd_ps(b0, x, w1);

                /* new_w1 = b1 * x - a1 * y + w2 */
                __m512 new_w1 = _mm512_fmadd_ps(b1, x, w2);
                new_w1 = _mm512_fnmadd_ps(a1, y, new_w1);

                /* new_w2 = b2 * x - a2 * y */
                __m512 new_w2 = _mm512_mul_ps(b2, x);
                new_w2 = _mm512_fnmadd_ps(a2, y, new_w2);

                _mm512_storeu_ps(&st->w1[i], new_w1);
                _mm512_storeu_ps(&st->w2[i], new_w2);

                x = y;  /* Output of this section is input to next */
            }

            _mm512_storeu_ps(&output[i], x);
        }

        /* Handle remaining positions with scalar code */
        for (size_t i = n_simd; i < n; i++) {
            float x = input[i];
            for (int s = 0; s < num_sections; s++) {
                const BiquadCoeffs *c = &coeffs[s];
                BiquadStateSoA *st = &state_soa[s];

                float w1 = st->w1[i];
                float w2 = st->w2[i];

                float y = c->b0 * x + w1;
                st->w1[i] = c->b1 * x - c->a1 * y + w2;
                st->w2[i] = c->b2 * x - c->a2 * y;

                x = y;
            }
            output[i] = x;
        }
    } else
#endif
    {
        /* Scalar fallback */
        BiquadState *state = filt->state;
        for (size_t i = 0; i < n; i++) {
            float x = input[i];
            BiquadState *pos_state = &state[i * num_sections];
            for (int s = 0; s < num_sections; s++) {
                x = biquad_process(&coeffs[s], &pos_state[s], x);
            }
            output[i] = x;
        }
    }

    filt->frames_processed++;
}

void iir_temporal_process_batch(IIRTemporalFilter *filt,
                                const float *input, float *output,
                                int batch_size) {
    const int num_sections = filt->coeffs.num_sections;
    const BiquadCoeffs *coeffs = filt->coeffs.sections;
    const size_t n = filt->num_positions;

#if USE_AVX512
    if (filt->use_simd) {
        BiquadStateSoA *state_soa = filt->state_soa;
        const size_t n_simd = n & ~(size_t)(SIMD_WIDTH - 1);

        /*
         * AVX-512 batch processing:
         * - Outer loop: positions in chunks of 16 (parallel with OpenMP)
         * - Middle loop: frames (sequential for IIR state)
         * - Inner loop: biquad sections (sequential cascade)
         *
         * NOTE: 4x loop unrolling was tried but provided no speedup.
         * The memory bandwidth is the bottleneck, not compute.
         */
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_simd; i += SIMD_WIDTH) {
            /* Process all frames for these 16 positions */
            for (int f = 0; f < batch_size; f++) {
                const size_t idx = (size_t)f * n + i;
                __m512 x = _mm512_loadu_ps(&input[idx]);

                for (int s = 0; s < num_sections; s++) {
                    const BiquadCoeffs *c = &coeffs[s];
                    BiquadStateSoA *st = &state_soa[s];

                    __m512 w1 = _mm512_loadu_ps(&st->w1[i]);
                    __m512 w2 = _mm512_loadu_ps(&st->w2[i]);

                    __m512 b0 = _mm512_set1_ps(c->b0);
                    __m512 b1 = _mm512_set1_ps(c->b1);
                    __m512 b2 = _mm512_set1_ps(c->b2);
                    __m512 a1 = _mm512_set1_ps(c->a1);
                    __m512 a2 = _mm512_set1_ps(c->a2);

                    __m512 y = _mm512_fmadd_ps(b0, x, w1);
                    __m512 new_w1 = _mm512_fmadd_ps(b1, x, w2);
                    new_w1 = _mm512_fnmadd_ps(a1, y, new_w1);
                    __m512 new_w2 = _mm512_mul_ps(b2, x);
                    new_w2 = _mm512_fnmadd_ps(a2, y, new_w2);

                    _mm512_storeu_ps(&st->w1[i], new_w1);
                    _mm512_storeu_ps(&st->w2[i], new_w2);

                    x = y;
                }

                _mm512_storeu_ps(&output[idx], x);
            }
        }

        /* Handle remaining positions with scalar code */
        for (size_t i = n_simd; i < n; i++) {
            for (int f = 0; f < batch_size; f++) {
                const size_t idx = (size_t)f * n + i;
                float x = input[idx];

                for (int s = 0; s < num_sections; s++) {
                    const BiquadCoeffs *c = &coeffs[s];
                    BiquadStateSoA *st = &state_soa[s];

                    float w1 = st->w1[i];
                    float w2 = st->w2[i];

                    float y = c->b0 * x + w1;
                    st->w1[i] = c->b1 * x - c->a1 * y + w2;
                    st->w2[i] = c->b2 * x - c->a2 * y;

                    x = y;
                }

                output[idx] = x;
            }
        }
    } else
#endif
    {
        /* Scalar fallback with OpenMP */
        BiquadState *state = filt->state;

        #pragma omp parallel for schedule(static)
        for (size_t pos = 0; pos < n; pos++) {
            BiquadState *pos_state = &state[pos * num_sections];

            for (int f = 0; f < batch_size; f++) {
                size_t idx = (size_t)f * n + pos;
                float x = input[idx];

                for (int s = 0; s < num_sections; s++) {
                    x = biquad_process(&coeffs[s], &pos_state[s], x);
                }

                output[idx] = x;
            }
        }
    }

    filt->frames_processed += batch_size;
}

bool iir_temporal_ready(const IIRTemporalFilter *filt) {
    return filt->frames_processed >= filt->warmup_frames;
}

void iir_temporal_preseed(IIRTemporalFilter *filt, const float *initial_coeffs) {
    if (filt->frames_processed > 0) {
        return;  /* Already processing, don't re-seed */
    }

    const int num_sections = filt->coeffs.num_sections;
    const BiquadCoeffs *coeffs = filt->coeffs.sections;
    const size_t n = filt->num_positions;
    const int warmup = filt->warmup_frames;

    /*
     * Feed the initial coefficients through the filter repeatedly
     * to bring it to steady state. This eliminates startup transient.
     */
#if USE_AVX512
    if (filt->use_simd) {
        BiquadStateSoA *state_soa = filt->state_soa;
        const size_t n_simd = n & ~(size_t)(SIMD_WIDTH - 1);

        for (int iter = 0; iter < warmup; iter++) {
            /* SIMD path */
            for (size_t i = 0; i < n_simd; i += SIMD_WIDTH) {
                __m512 x = _mm512_loadu_ps(&initial_coeffs[i]);

                for (int s = 0; s < num_sections; s++) {
                    const BiquadCoeffs *c = &coeffs[s];
                    BiquadStateSoA *st = &state_soa[s];

                    __m512 w1 = _mm512_loadu_ps(&st->w1[i]);
                    __m512 w2 = _mm512_loadu_ps(&st->w2[i]);

                    __m512 b0 = _mm512_set1_ps(c->b0);
                    __m512 b1 = _mm512_set1_ps(c->b1);
                    __m512 b2 = _mm512_set1_ps(c->b2);
                    __m512 a1 = _mm512_set1_ps(c->a1);
                    __m512 a2 = _mm512_set1_ps(c->a2);

                    __m512 y = _mm512_fmadd_ps(b0, x, w1);
                    __m512 new_w1 = _mm512_fmadd_ps(b1, x, w2);
                    new_w1 = _mm512_fnmadd_ps(a1, y, new_w1);
                    __m512 new_w2 = _mm512_mul_ps(b2, x);
                    new_w2 = _mm512_fnmadd_ps(a2, y, new_w2);

                    _mm512_storeu_ps(&st->w1[i], new_w1);
                    _mm512_storeu_ps(&st->w2[i], new_w2);

                    x = y;
                }
            }

            /* Scalar remainder */
            for (size_t i = n_simd; i < n; i++) {
                float x = initial_coeffs[i];
                for (int s = 0; s < num_sections; s++) {
                    const BiquadCoeffs *c = &coeffs[s];
                    BiquadStateSoA *st = &state_soa[s];

                    float w1 = st->w1[i];
                    float w2 = st->w2[i];

                    float y = c->b0 * x + w1;
                    st->w1[i] = c->b1 * x - c->a1 * y + w2;
                    st->w2[i] = c->b2 * x - c->a2 * y;

                    x = y;
                }
            }
        }
    } else
#endif
    {
        /* Scalar fallback */
        BiquadState *state = filt->state;

        for (int iter = 0; iter < warmup; iter++) {
            for (size_t i = 0; i < n; i++) {
                float x = initial_coeffs[i];
                BiquadState *pos_state = &state[i * num_sections];

                for (int s = 0; s < num_sections; s++) {
                    const BiquadCoeffs *c = &coeffs[s];
                    BiquadState *st = &pos_state[s];

                    float y = c->b0 * x + st->w1;
                    st->w1 = c->b1 * x - c->a1 * y + st->w2;
                    st->w2 = c->b2 * x - c->a2 * y;

                    x = y;
                }
            }
        }
    }

    /* Mark as warmed up */
    filt->frames_processed = warmup;
}
