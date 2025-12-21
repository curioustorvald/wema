/*
 * WEMA - Dual-Tree Complex Wavelet Transform
 *
 * Simplified implementation using Haar wavelets for spatial decomposition.
 * Creates pseudo-complex coefficients using horizontal/vertical separation.
 */

#include "dtcwt.h"
#include "dtcwt_filters.h"
#include "complex_math.h"
#include "alloc.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/*============================================================================
 * Simple Haar DWT (guaranteed perfect reconstruction)
 *===========================================================================*/

/* 2D Haar forward transform - one level */
static void haar_2d_forward(const float *input, int width, int height,
                            float *ll, float *lh, float *hl, float *hh,
                            int out_w, int out_h) {
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            int y2 = y * 2;
            int x2 = x * 2;

            /* Get 2x2 block with boundary handling */
            float a = input[y2 * width + x2];
            float b = (x2 + 1 < width) ? input[y2 * width + x2 + 1] : a;
            float c = (y2 + 1 < height) ? input[(y2 + 1) * width + x2] : a;
            float d = (x2 + 1 < width && y2 + 1 < height) ?
                      input[(y2 + 1) * width + x2 + 1] :
                      ((y2 + 1 < height) ? c : b);

            /* 2D Haar transform */
            ll[y * out_w + x] = (a + b + c + d) * 0.25f;  /* Average */
            lh[y * out_w + x] = (a + b - c - d) * 0.25f;  /* Horizontal edge */
            hl[y * out_w + x] = (a - b + c - d) * 0.25f;  /* Vertical edge */
            hh[y * out_w + x] = (a - b - c + d) * 0.25f;  /* Diagonal */
        }
    }
}

/* 2D Haar inverse transform - one level */
static void haar_2d_inverse(const float *ll, const float *lh,
                            const float *hl, const float *hh,
                            int in_w, int in_h,
                            float *output, int out_w, int out_h) {
    for (int y = 0; y < in_h; y++) {
        for (int x = 0; x < in_w; x++) {
            float l = ll[y * in_w + x];
            float h1 = lh[y * in_w + x];
            float h2 = hl[y * in_w + x];
            float h3 = hh[y * in_w + x];

            /* Inverse 2D Haar */
            float a = l + h1 + h2 + h3;
            float b = l + h1 - h2 - h3;
            float c = l - h1 + h2 - h3;
            float d = l - h1 - h2 + h3;

            int y2 = y * 2;
            int x2 = x * 2;

            if (y2 < out_h && x2 < out_w)
                output[y2 * out_w + x2] = a;
            if (y2 < out_h && x2 + 1 < out_w)
                output[y2 * out_w + x2 + 1] = b;
            if (y2 + 1 < out_h && x2 < out_w)
                output[(y2 + 1) * out_w + x2] = c;
            if (y2 + 1 < out_h && x2 + 1 < out_w)
                output[(y2 + 1) * out_w + x2 + 1] = d;
        }
    }
}

/*============================================================================
 * Create complex coefficients from real subbands
 *
 * Use LH (horizontal) and HL (vertical) to create oriented complex coefficients.
 * This is a simplified version that creates 6 orientations from the subbands.
 *===========================================================================*/

/* Store real subbands directly in complex format for now */
static void create_complex_subbands(const float *lh, const float *hl, const float *hh,
                                    int w, int h, Subband *subbands) {
    int n = w * h;

    for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
        subbands[o].width = w;
        subbands[o].height = h;
    }

    for (int i = 0; i < n; i++) {
        /* Store subbands directly - real in .re, zero in .im */
        /* This allows phase extraction to work on edge magnitudes */
        subbands[0].coeffs[i] = cmplx(lh[i], 0.001f * lh[i]); /* horizontal */
        subbands[1].coeffs[i] = cmplx(hh[i], 0.001f * hh[i]); /* diagonal */
        subbands[2].coeffs[i] = cmplx(hl[i], 0.001f * hl[i]); /* vertical */
        subbands[3].coeffs[i] = subbands[2].coeffs[i];
        subbands[4].coeffs[i] = subbands[1].coeffs[i];
        subbands[5].coeffs[i] = subbands[0].coeffs[i];
    }
}

static void extract_real_subbands(const Subband *subbands,
                                  float *lh, float *hl, float *hh,
                                  int w, int h) {
    int n = w * h;

    for (int i = 0; i < n; i++) {
        /* Extract real parts */
        lh[i] = subbands[0].coeffs[i].re;
        hl[i] = subbands[2].coeffs[i].re;
        hh[i] = subbands[1].coeffs[i].re;
    }
}

/*============================================================================
 * Public API
 *===========================================================================*/

int dtcwt_compute_levels(int width, int height) {
    int min_dim = (width < height) ? width : height;
    int levels = 0;
    while (min_dim >= 16 && levels < WEMA_MAX_LEVELS) {
        min_dim /= 2;
        levels++;
    }
    return levels > 0 ? levels : 1;
}

int dtcwt_init(DTCWTCoeffs *coeffs, int width, int height, int num_levels) {
    memset(coeffs, 0, sizeof(*coeffs));

    coeffs->orig_width = width;
    coeffs->orig_height = height;
    coeffs->num_levels = num_levels;

    int w = width;
    int h = height;

    for (int lev = 0; lev < num_levels; lev++) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;

        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            coeffs->subbands[lev][o].width = w;
            coeffs->subbands[lev][o].height = h;
            coeffs->subbands[lev][o].coeffs = mem_calloc(w * h, sizeof(Complex));
            if (!coeffs->subbands[lev][o].coeffs) {
                dtcwt_free(coeffs);
                return -1;
            }
        }
    }

    /* Lowpass residual */
    coeffs->lowpass_w = w;
    coeffs->lowpass_h = h;
    coeffs->lowpass = mem_calloc(w * h, sizeof(float));
    if (!coeffs->lowpass) {
        dtcwt_free(coeffs);
        return -1;
    }

    return 0;
}

void dtcwt_free(DTCWTCoeffs *coeffs) {
    for (int lev = 0; lev < WEMA_MAX_LEVELS; lev++) {
        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            mem_free(coeffs->subbands[lev][o].coeffs);
            coeffs->subbands[lev][o].coeffs = NULL;
        }
    }
    mem_free(coeffs->lowpass);
    coeffs->lowpass = NULL;
}

void dtcwt_forward(const float *image, int width, int height,
                   DTCWTCoeffs *coeffs) {
    /* Allocate work buffers */
    size_t max_size = (size_t)width * height;
    float *current = mem_alloc(max_size * sizeof(float));
    float *ll = mem_alloc(max_size * sizeof(float));
    float *lh = mem_alloc(max_size * sizeof(float));
    float *hl = mem_alloc(max_size * sizeof(float));
    float *hh = mem_alloc(max_size * sizeof(float));

    if (!current || !ll || !lh || !hl || !hh) {
        mem_free(current);
        mem_free(ll);
        mem_free(lh);
        mem_free(hl);
        mem_free(hh);
        return;
    }

    /* Copy input */
    memcpy(current, image, width * height * sizeof(float));

    int cur_w = width;
    int cur_h = height;

    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        int out_w = (cur_w + 1) / 2;
        int out_h = (cur_h + 1) / 2;

        /* Haar forward transform */
        haar_2d_forward(current, cur_w, cur_h, ll, lh, hl, hh, out_w, out_h);

        /* Create complex subbands */
        create_complex_subbands(lh, hl, hh, out_w, out_h, coeffs->subbands[lev]);

        /* Copy lowpass for next level */
        memcpy(current, ll, out_w * out_h * sizeof(float));
        cur_w = out_w;
        cur_h = out_h;
    }

    /* Store final lowpass */
    memcpy(coeffs->lowpass, current, cur_w * cur_h * sizeof(float));

    mem_free(current);
    mem_free(ll);
    mem_free(lh);
    mem_free(hl);
    mem_free(hh);
}

void dtcwt_inverse(const DTCWTCoeffs *coeffs, float *image) {
    /* Allocate work buffers */
    size_t max_size = (size_t)coeffs->orig_width * coeffs->orig_height;
    float *current = mem_alloc(max_size * sizeof(float));
    float *lh = mem_alloc(max_size * sizeof(float));
    float *hl = mem_alloc(max_size * sizeof(float));
    float *hh = mem_alloc(max_size * sizeof(float));
    float *output = mem_alloc(max_size * sizeof(float));

    if (!current || !lh || !hl || !hh || !output) {
        mem_free(current);
        mem_free(lh);
        mem_free(hl);
        mem_free(hh);
        mem_free(output);
        return;
    }

    /* Precompute the input dimensions at each level during forward transform.
     * This is needed because (w+1)/2 * 2 != w when w is odd. */
    int level_w[WEMA_MAX_LEVELS + 1];
    int level_h[WEMA_MAX_LEVELS + 1];
    level_w[0] = coeffs->orig_width;
    level_h[0] = coeffs->orig_height;
    for (int lev = 1; lev <= coeffs->num_levels; lev++) {
        level_w[lev] = (level_w[lev - 1] + 1) / 2;
        level_h[lev] = (level_h[lev - 1] + 1) / 2;
    }

    /* Start with the coarsest lowpass */
    int cur_w = coeffs->lowpass_w;
    int cur_h = coeffs->lowpass_h;
    memcpy(current, coeffs->lowpass, cur_w * cur_h * sizeof(float));

    /* Reconstruct from coarsest to finest */
    for (int lev = coeffs->num_levels - 1; lev >= 0; lev--) {
        int sub_w = coeffs->subbands[lev][0].width;
        int sub_h = coeffs->subbands[lev][0].height;

        /* Extract real subbands from complex */
        extract_real_subbands(coeffs->subbands[lev], lh, hl, hh, sub_w, sub_h);

        /* Output size is the input size to this level during forward transform */
        int out_w = level_w[lev];
        int out_h = level_h[lev];

        /* Haar inverse transform */
        haar_2d_inverse(current, lh, hl, hh, sub_w, sub_h, output, out_w, out_h);

        /* Prepare for next level */
        memcpy(current, output, out_w * out_h * sizeof(float));
        cur_w = out_w;
        cur_h = out_h;
    }

    /* Copy final result */
    memcpy(image, current, coeffs->orig_width * coeffs->orig_height * sizeof(float));

    mem_free(current);
    mem_free(lh);
    mem_free(hl);
    mem_free(hh);
    mem_free(output);
}

size_t dtcwt_num_positions(const DTCWTCoeffs *coeffs) {
    size_t total = 0;
    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            total += (size_t)coeffs->subbands[lev][o].width *
                     coeffs->subbands[lev][o].height;
        }
    }
    return total;
}
