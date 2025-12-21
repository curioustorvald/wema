/*
 * WEMA - Dual-Tree Complex Wavelet Transform
 *
 * Proper implementation using Kingsbury's Q-shift filters.
 * Two parallel trees with half-sample offset create complex wavelets
 * with meaningful phase for motion amplification.
 */

#include "dtcwt.h"
#include "dtcwt_filters.h"
#include "complex_math.h"
#include "alloc.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/*============================================================================
 * 1D Convolution with decimation (analysis) and interpolation (synthesis)
 *===========================================================================*/

/* Symmetric boundary extension for convolution */
static inline int reflect(int i, int n) {
    if (i < 0) return -i - 1;
    if (i >= n) return 2 * n - i - 1;
    return i;
}

/* 1D convolution with decimation by 2 (analysis) */
static void conv_decimate(const float *input, int in_len,
                          const float *filter, int filt_len,
                          float *output, int out_len) {
    int half = filt_len / 2;

    for (int i = 0; i < out_len; i++) {
        float sum = 0.0f;
        int in_pos = 2 * i;

        for (int k = 0; k < filt_len; k++) {
            int idx = reflect(in_pos - half + k, in_len);
            sum += filter[k] * input[idx];
        }
        output[i] = sum;
    }
}

/* 1D interpolation (upsample by 2) with convolution (synthesis) */
static void interp_conv(const float *input, int in_len,
                        const float *filter, int filt_len,
                        float *output, int out_len) {
    int half = filt_len / 2;

    /* Clear output */
    memset(output, 0, out_len * sizeof(float));

    for (int i = 0; i < in_len; i++) {
        float val = input[i];
        int out_pos = 2 * i;

        for (int k = 0; k < filt_len; k++) {
            int idx = out_pos - half + k;
            if (idx >= 0 && idx < out_len) {
                output[idx] += filter[k] * val;
            }
        }
    }
}

/*============================================================================
 * 2D Separable Wavelet Transform (one level, one tree)
 *===========================================================================*/

/* 2D analysis: apply 1D filters along rows then columns */
static void dwt2d_analysis(const float *input, int width, int height,
                           const float *lo_filt, const float *hi_filt, int filt_len,
                           float *ll, float *lh, float *hl, float *hh,
                           int out_w, int out_h) {
    /* Temporary buffer for row filtering */
    float *temp_lo = mem_alloc((size_t)out_w * height * sizeof(float));
    float *temp_hi = mem_alloc((size_t)out_w * height * sizeof(float));
    float *row_in = mem_alloc(width * sizeof(float));
    float *row_out = mem_alloc(out_w * sizeof(float));

    if (!temp_lo || !temp_hi || !row_in || !row_out) {
        mem_free(temp_lo);
        mem_free(temp_hi);
        mem_free(row_in);
        mem_free(row_out);
        return;
    }

    /* Filter along rows first */
    for (int y = 0; y < height; y++) {
        /* Extract row */
        for (int x = 0; x < width; x++) {
            row_in[x] = input[y * width + x];
        }

        /* Lowpass along row */
        conv_decimate(row_in, width, lo_filt, filt_len, row_out, out_w);
        for (int x = 0; x < out_w; x++) {
            temp_lo[y * out_w + x] = row_out[x];
        }

        /* Highpass along row */
        conv_decimate(row_in, width, hi_filt, filt_len, row_out, out_w);
        for (int x = 0; x < out_w; x++) {
            temp_hi[y * out_w + x] = row_out[x];
        }
    }

    /* Now filter along columns */
    float *col_in = mem_alloc(height * sizeof(float));
    float *col_out = mem_alloc(out_h * sizeof(float));

    if (!col_in || !col_out) {
        mem_free(temp_lo);
        mem_free(temp_hi);
        mem_free(row_in);
        mem_free(row_out);
        mem_free(col_in);
        mem_free(col_out);
        return;
    }

    /* LL: lowpass on temp_lo columns */
    for (int x = 0; x < out_w; x++) {
        for (int y = 0; y < height; y++) {
            col_in[y] = temp_lo[y * out_w + x];
        }
        conv_decimate(col_in, height, lo_filt, filt_len, col_out, out_h);
        for (int y = 0; y < out_h; y++) {
            ll[y * out_w + x] = col_out[y];
        }
    }

    /* LH: highpass on temp_lo columns (horizontal edges) */
    for (int x = 0; x < out_w; x++) {
        for (int y = 0; y < height; y++) {
            col_in[y] = temp_lo[y * out_w + x];
        }
        conv_decimate(col_in, height, hi_filt, filt_len, col_out, out_h);
        for (int y = 0; y < out_h; y++) {
            lh[y * out_w + x] = col_out[y];
        }
    }

    /* HL: lowpass on temp_hi columns (vertical edges) */
    for (int x = 0; x < out_w; x++) {
        for (int y = 0; y < height; y++) {
            col_in[y] = temp_hi[y * out_w + x];
        }
        conv_decimate(col_in, height, lo_filt, filt_len, col_out, out_h);
        for (int y = 0; y < out_h; y++) {
            hl[y * out_w + x] = col_out[y];
        }
    }

    /* HH: highpass on temp_hi columns (diagonal edges) */
    for (int x = 0; x < out_w; x++) {
        for (int y = 0; y < height; y++) {
            col_in[y] = temp_hi[y * out_w + x];
        }
        conv_decimate(col_in, height, hi_filt, filt_len, col_out, out_h);
        for (int y = 0; y < out_h; y++) {
            hh[y * out_w + x] = col_out[y];
        }
    }

    mem_free(temp_lo);
    mem_free(temp_hi);
    mem_free(row_in);
    mem_free(row_out);
    mem_free(col_in);
    mem_free(col_out);
}

/* 2D synthesis: apply filters along columns then rows */
static void dwt2d_synthesis(const float *ll, const float *lh,
                            const float *hl, const float *hh,
                            int in_w, int in_h,
                            const float *lo_filt, const float *hi_filt, int filt_len,
                            float *output, int out_w, int out_h) {
    /* Temporary buffers */
    float *temp_lo = mem_alloc((size_t)in_w * out_h * sizeof(float));
    float *temp_hi = mem_alloc((size_t)in_w * out_h * sizeof(float));

    if (!temp_lo || !temp_hi) {
        mem_free(temp_lo);
        mem_free(temp_hi);
        return;
    }

    float *col_lo = mem_alloc(in_h * sizeof(float));
    float *col_hi = mem_alloc(in_h * sizeof(float));
    float *col_out = mem_alloc(out_h * sizeof(float));

    if (!col_lo || !col_hi || !col_out) {
        mem_free(temp_lo);
        mem_free(temp_hi);
        mem_free(col_lo);
        mem_free(col_hi);
        mem_free(col_out);
        return;
    }

    /* Synthesize columns: combine LL+LH -> temp_lo, HL+HH -> temp_hi */
    for (int x = 0; x < in_w; x++) {
        /* Extract LL and LH columns */
        for (int y = 0; y < in_h; y++) {
            col_lo[y] = ll[y * in_w + x];
            col_hi[y] = lh[y * in_w + x];
        }

        /* Interpolate and filter */
        float *lo_interp = mem_alloc(out_h * sizeof(float));
        float *hi_interp = mem_alloc(out_h * sizeof(float));

        interp_conv(col_lo, in_h, lo_filt, filt_len, lo_interp, out_h);
        interp_conv(col_hi, in_h, hi_filt, filt_len, hi_interp, out_h);

        for (int y = 0; y < out_h; y++) {
            temp_lo[y * in_w + x] = lo_interp[y] + hi_interp[y];
        }

        mem_free(lo_interp);
        mem_free(hi_interp);

        /* Extract HL and HH columns */
        for (int y = 0; y < in_h; y++) {
            col_lo[y] = hl[y * in_w + x];
            col_hi[y] = hh[y * in_w + x];
        }

        lo_interp = mem_alloc(out_h * sizeof(float));
        hi_interp = mem_alloc(out_h * sizeof(float));

        interp_conv(col_lo, in_h, lo_filt, filt_len, lo_interp, out_h);
        interp_conv(col_hi, in_h, hi_filt, filt_len, hi_interp, out_h);

        for (int y = 0; y < out_h; y++) {
            temp_hi[y * in_w + x] = lo_interp[y] + hi_interp[y];
        }

        mem_free(lo_interp);
        mem_free(hi_interp);
    }

    mem_free(col_lo);
    mem_free(col_hi);
    mem_free(col_out);

    /* Synthesize rows: combine temp_lo + temp_hi -> output */
    float *row_lo = mem_alloc(in_w * sizeof(float));
    float *row_hi = mem_alloc(in_w * sizeof(float));

    if (!row_lo || !row_hi) {
        mem_free(temp_lo);
        mem_free(temp_hi);
        mem_free(row_lo);
        mem_free(row_hi);
        return;
    }

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < in_w; x++) {
            row_lo[x] = temp_lo[y * in_w + x];
            row_hi[x] = temp_hi[y * in_w + x];
        }

        float *lo_interp = mem_alloc(out_w * sizeof(float));
        float *hi_interp = mem_alloc(out_w * sizeof(float));

        interp_conv(row_lo, in_w, lo_filt, filt_len, lo_interp, out_w);
        interp_conv(row_hi, in_w, hi_filt, filt_len, hi_interp, out_w);

        for (int x = 0; x < out_w; x++) {
            output[y * out_w + x] = lo_interp[x] + hi_interp[x];
        }

        mem_free(lo_interp);
        mem_free(hi_interp);
    }

    mem_free(temp_lo);
    mem_free(temp_hi);
    mem_free(row_lo);
    mem_free(row_hi);
}

/*============================================================================
 * Complex subband creation from dual trees
 *
 * The 6 complex subbands at each level are created by combining the
 * subbands from Tree A and Tree B to get directionally selective wavelets.
 *
 * Orientations (following Selesnick's paper):
 *   0: +15° -> (LH_A - LH_B) + j(LH_A + LH_B) ... simplified to LH_A + j*LH_B
 *   1: +45° -> HH_A + j*HH_B
 *   2: +75° -> HL_A + j*HL_B
 *   3: -75° -> HL_A - j*HL_B
 *   4: -45° -> HH_A - j*HH_B
 *   5: -15° -> LH_A - j*LH_B
 *===========================================================================*/

static void create_complex_subbands_from_trees(
    const float *lh_a, const float *hl_a, const float *hh_a,
    const float *lh_b, const float *hl_b, const float *hh_b,
    int w, int h, Subband *subbands) {

    int n = w * h;

    for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
        subbands[o].width = w;
        subbands[o].height = h;
    }

    for (int i = 0; i < n; i++) {
        /* Create 6 complex orientations from the two trees */
        /* The factor of 1/sqrt(2) normalizes the combination */
        float s = 0.70710678118f;  /* 1/sqrt(2) */

        /* +15°: combines horizontal subbands */
        subbands[0].coeffs[i] = cmplx(s * lh_a[i], s * lh_b[i]);

        /* +45°: combines diagonal subbands */
        subbands[1].coeffs[i] = cmplx(s * hh_a[i], s * hh_b[i]);

        /* +75°: combines vertical subbands */
        subbands[2].coeffs[i] = cmplx(s * hl_a[i], s * hl_b[i]);

        /* -75°: conjugate of +75° */
        subbands[3].coeffs[i] = cmplx(s * hl_a[i], -s * hl_b[i]);

        /* -45°: conjugate of +45° */
        subbands[4].coeffs[i] = cmplx(s * hh_a[i], -s * hh_b[i]);

        /* -15°: conjugate of +15° */
        subbands[5].coeffs[i] = cmplx(s * lh_a[i], -s * lh_b[i]);
    }
}

static void extract_trees_from_complex(const Subband *subbands,
                                       float *lh_a, float *hl_a, float *hh_a,
                                       float *lh_b, float *hl_b, float *hh_b,
                                       int w, int h) {
    int n = w * h;
    float s = 0.70710678118f;  /* 1/sqrt(2) */

    for (int i = 0; i < n; i++) {
        /* Extract Tree A (real parts) and Tree B (imag parts) */
        /* Average positive and negative orientations for reconstruction */

        /* LH: average of orientations 0 and 5 */
        lh_a[i] = (subbands[0].coeffs[i].re + subbands[5].coeffs[i].re) / s;
        lh_b[i] = (subbands[0].coeffs[i].im - subbands[5].coeffs[i].im) / s;

        /* HH: average of orientations 1 and 4 */
        hh_a[i] = (subbands[1].coeffs[i].re + subbands[4].coeffs[i].re) / s;
        hh_b[i] = (subbands[1].coeffs[i].im - subbands[4].coeffs[i].im) / s;

        /* HL: average of orientations 2 and 3 */
        hl_a[i] = (subbands[2].coeffs[i].re + subbands[3].coeffs[i].re) / s;
        hl_b[i] = (subbands[2].coeffs[i].im - subbands[3].coeffs[i].im) / s;
    }
}

/*============================================================================
 * Public API
 *===========================================================================*/

int dtcwt_compute_levels(int width, int height) {
    int min_dim = (width < height) ? width : height;
    int levels = 0;
    while (min_dim >= 32 && levels < WEMA_MAX_LEVELS) {
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
    /* Allocate work buffers for both trees */
    size_t max_size = (size_t)width * height;

    float *current_a = mem_alloc(max_size * sizeof(float));
    float *current_b = mem_alloc(max_size * sizeof(float));
    float *ll_a = mem_alloc(max_size * sizeof(float));
    float *lh_a = mem_alloc(max_size * sizeof(float));
    float *hl_a = mem_alloc(max_size * sizeof(float));
    float *hh_a = mem_alloc(max_size * sizeof(float));
    float *ll_b = mem_alloc(max_size * sizeof(float));
    float *lh_b = mem_alloc(max_size * sizeof(float));
    float *hl_b = mem_alloc(max_size * sizeof(float));
    float *hh_b = mem_alloc(max_size * sizeof(float));

    if (!current_a || !current_b || !ll_a || !lh_a || !hl_a || !hh_a ||
        !ll_b || !lh_b || !hl_b || !hh_b) {
        mem_free(current_a); mem_free(current_b);
        mem_free(ll_a); mem_free(lh_a); mem_free(hl_a); mem_free(hh_a);
        mem_free(ll_b); mem_free(lh_b); mem_free(hl_b); mem_free(hh_b);
        return;
    }

    /* Both trees start with the same input */
    memcpy(current_a, image, width * height * sizeof(float));
    memcpy(current_b, image, width * height * sizeof(float));

    int cur_w = width;
    int cur_h = height;

    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        int out_w = (cur_w + 1) / 2;
        int out_h = (cur_h + 1) / 2;

        /* Select filters based on level */
        const float *lo_a, *hi_a, *lo_b, *hi_b;
        int filt_len;

        if (lev == 0) {
            /* First level: near-symmetric filters */
            lo_a = H0A_L1; hi_a = H1A_L1;
            lo_b = H0B_L1; hi_b = H1B_L1;
            filt_len = DTCWT_L1_LEN;
        } else {
            /* Subsequent levels: Q-shift filters */
            lo_a = H0A_Q; hi_a = H1A_Q;
            lo_b = H0B_Q; hi_b = H1B_Q;
            filt_len = DTCWT_Q_LEN;
        }

        /* Apply 2D DWT to Tree A */
        dwt2d_analysis(current_a, cur_w, cur_h, lo_a, hi_a, filt_len,
                       ll_a, lh_a, hl_a, hh_a, out_w, out_h);

        /* Apply 2D DWT to Tree B */
        dwt2d_analysis(current_b, cur_w, cur_h, lo_b, hi_b, filt_len,
                       ll_b, lh_b, hl_b, hh_b, out_w, out_h);

        /* Create complex subbands from the two trees */
        create_complex_subbands_from_trees(lh_a, hl_a, hh_a,
                                           lh_b, hl_b, hh_b,
                                           out_w, out_h,
                                           coeffs->subbands[lev]);

        /* Update current buffers with lowpass for next level */
        memcpy(current_a, ll_a, out_w * out_h * sizeof(float));
        memcpy(current_b, ll_b, out_w * out_h * sizeof(float));
        cur_w = out_w;
        cur_h = out_h;
    }

    /* Store final lowpass (average of both trees) */
    for (int i = 0; i < cur_w * cur_h; i++) {
        coeffs->lowpass[i] = 0.5f * (current_a[i] + current_b[i]);
    }

    mem_free(current_a); mem_free(current_b);
    mem_free(ll_a); mem_free(lh_a); mem_free(hl_a); mem_free(hh_a);
    mem_free(ll_b); mem_free(lh_b); mem_free(hl_b); mem_free(hh_b);
}

void dtcwt_inverse(const DTCWTCoeffs *coeffs, float *image) {
    size_t max_size = (size_t)coeffs->orig_width * coeffs->orig_height;

    float *current_a = mem_alloc(max_size * sizeof(float));
    float *current_b = mem_alloc(max_size * sizeof(float));
    float *ll_a = mem_alloc(max_size * sizeof(float));
    float *lh_a = mem_alloc(max_size * sizeof(float));
    float *hl_a = mem_alloc(max_size * sizeof(float));
    float *hh_a = mem_alloc(max_size * sizeof(float));
    float *ll_b = mem_alloc(max_size * sizeof(float));
    float *lh_b = mem_alloc(max_size * sizeof(float));
    float *hl_b = mem_alloc(max_size * sizeof(float));
    float *hh_b = mem_alloc(max_size * sizeof(float));
    float *output_a = mem_alloc(max_size * sizeof(float));
    float *output_b = mem_alloc(max_size * sizeof(float));

    if (!current_a || !current_b || !ll_a || !lh_a || !hl_a || !hh_a ||
        !ll_b || !lh_b || !hl_b || !hh_b || !output_a || !output_b) {
        mem_free(current_a); mem_free(current_b);
        mem_free(ll_a); mem_free(lh_a); mem_free(hl_a); mem_free(hh_a);
        mem_free(ll_b); mem_free(lh_b); mem_free(hl_b); mem_free(hh_b);
        mem_free(output_a); mem_free(output_b);
        return;
    }

    /* Precompute level dimensions */
    int level_w[WEMA_MAX_LEVELS + 1];
    int level_h[WEMA_MAX_LEVELS + 1];
    level_w[0] = coeffs->orig_width;
    level_h[0] = coeffs->orig_height;
    for (int lev = 1; lev <= coeffs->num_levels; lev++) {
        level_w[lev] = (level_w[lev - 1] + 1) / 2;
        level_h[lev] = (level_h[lev - 1] + 1) / 2;
    }

    /* Start with lowpass (same for both trees) */
    int cur_w = coeffs->lowpass_w;
    int cur_h = coeffs->lowpass_h;
    memcpy(current_a, coeffs->lowpass, cur_w * cur_h * sizeof(float));
    memcpy(current_b, coeffs->lowpass, cur_w * cur_h * sizeof(float));

    /* Reconstruct from coarsest to finest */
    for (int lev = coeffs->num_levels - 1; lev >= 0; lev--) {
        int sub_w = coeffs->subbands[lev][0].width;
        int sub_h = coeffs->subbands[lev][0].height;
        int out_w = level_w[lev];
        int out_h = level_h[lev];

        /* Extract trees from complex subbands */
        extract_trees_from_complex(coeffs->subbands[lev],
                                   lh_a, hl_a, hh_a,
                                   lh_b, hl_b, hh_b,
                                   sub_w, sub_h);

        /* Current lowpass becomes LL for this level */
        memcpy(ll_a, current_a, sub_w * sub_h * sizeof(float));
        memcpy(ll_b, current_b, sub_w * sub_h * sizeof(float));

        /* Select synthesis filters */
        const float *lo_a, *hi_a, *lo_b, *hi_b;
        int filt_len;

        if (lev == 0) {
            lo_a = G0A_L1; hi_a = G1A_L1;
            lo_b = G0B_L1; hi_b = G1B_L1;
            filt_len = DTCWT_L1_LEN;
        } else {
            lo_a = G0A_Q; hi_a = G1A_Q;
            lo_b = G0B_Q; hi_b = G1B_Q;
            filt_len = DTCWT_Q_LEN;
        }

        /* Inverse 2D DWT for Tree A */
        dwt2d_synthesis(ll_a, lh_a, hl_a, hh_a, sub_w, sub_h,
                        lo_a, hi_a, filt_len, output_a, out_w, out_h);

        /* Inverse 2D DWT for Tree B */
        dwt2d_synthesis(ll_b, lh_b, hl_b, hh_b, sub_w, sub_h,
                        lo_b, hi_b, filt_len, output_b, out_w, out_h);

        /* Update current with reconstructed outputs */
        memcpy(current_a, output_a, out_w * out_h * sizeof(float));
        memcpy(current_b, output_b, out_w * out_h * sizeof(float));
        cur_w = out_w;
        cur_h = out_h;
    }

    /* Final output: average of both trees */
    for (int i = 0; i < coeffs->orig_width * coeffs->orig_height; i++) {
        image[i] = 0.5f * (current_a[i] + current_b[i]);
    }

    mem_free(current_a); mem_free(current_b);
    mem_free(ll_a); mem_free(lh_a); mem_free(hl_a); mem_free(hh_a);
    mem_free(ll_b); mem_free(lh_b); mem_free(hl_b); mem_free(hh_b);
    mem_free(output_a); mem_free(output_b);
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
