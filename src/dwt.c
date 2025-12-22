/*
 * WEMA - 2D Discrete Wavelet Transform
 *
 * Implementation using CDF 9/7 biorthogonal wavelets via lifting scheme.
 * Uses standard dyadic decomposition with uniform subband sizes.
 */

#include "dwt.h"
#include "alloc.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/*============================================================================
 * CDF 9/7 Lifting Coefficients
 *===========================================================================*/

#define ALPHA -1.586134342059924f
#define BETA  -0.052980118572961f
#define GAMMA  0.882911075530934f
#define DELTA  0.443506852043971f
#define K      1.149604398860241f
#define K_INV  (1.0f / K)

/*============================================================================
 * 1D CDF 9/7 Lifting Implementation (Vectorization-Optimized)
 *
 * Uses split even/odd arrays for stride-1 access patterns.
 * temp layout: [even samples (half)] [odd samples (odd_n)]
 *===========================================================================*/

static void cdf97_fwd_1d(float * restrict data, int n, float * restrict temp) {
    if (n < 2) return;

    const int half = (n + 1) / 2;
    const int odd_n = n / 2;

    /* Split into even/odd - stride-1 writes */
    float * restrict even = temp;
    float * restrict odd = temp + half;

    for (int i = 0; i < odd_n; i++) {
        even[i] = data[2 * i];
        odd[i] = data[2 * i + 1];
    }
    if (half > odd_n) {
        even[odd_n] = data[2 * odd_n];
    }

    /* Predict 1: odd[i] += ALPHA * (even[i] + even[i+1]) */
    for (int i = 0; i < odd_n - 1; i++) {
        odd[i] += ALPHA * (even[i] + even[i + 1]);
    }
    if (odd_n > 0) {
        /* Boundary: reflect */
        odd[odd_n - 1] += ALPHA * (even[odd_n - 1] + even[half > odd_n ? odd_n : odd_n - 1]);
    }

    /* Update 1: even[i] += BETA * (odd[i-1] + odd[i]) */
    if (odd_n > 0) {
        even[0] += BETA * (odd[0] + odd[0]);  /* Reflect left boundary */
        for (int i = 1; i < half; i++) {
            int left = i - 1;
            int right = (i < odd_n) ? i : odd_n - 1;
            even[i] += BETA * (odd[left] + odd[right]);
        }
    }

    /* Predict 2: odd[i] += GAMMA * (even[i] + even[i+1]) */
    for (int i = 0; i < odd_n - 1; i++) {
        odd[i] += GAMMA * (even[i] + even[i + 1]);
    }
    if (odd_n > 0) {
        odd[odd_n - 1] += GAMMA * (even[odd_n - 1] + even[half > odd_n ? odd_n : odd_n - 1]);
    }

    /* Update 2: even[i] += DELTA * (odd[i-1] + odd[i]) */
    if (odd_n > 0) {
        even[0] += DELTA * (odd[0] + odd[0]);
        for (int i = 1; i < half; i++) {
            int left = i - 1;
            int right = (i < odd_n) ? i : odd_n - 1;
            even[i] += DELTA * (odd[left] + odd[right]);
        }
    }

    /* Scale and output - vectorizable */
    for (int i = 0; i < half; i++) {
        data[i] = even[i] * K_INV;
    }
    for (int i = 0; i < odd_n; i++) {
        data[half + i] = odd[i] * K;
    }
}

static void cdf97_inv_1d(float * restrict data, int n, float * restrict temp) {
    if (n < 2) return;

    const int half = (n + 1) / 2;
    const int odd_n = n / 2;

    float * restrict even = temp;
    float * restrict odd = temp + half;

    /* Unpack and unscale - vectorizable */
    for (int i = 0; i < half; i++) {
        even[i] = data[i] * K;
    }
    for (int i = 0; i < odd_n; i++) {
        odd[i] = data[half + i] * K_INV;
    }

    /* Inverse update 2 */
    if (odd_n > 0) {
        even[0] -= DELTA * (odd[0] + odd[0]);
        for (int i = 1; i < half; i++) {
            int left = i - 1;
            int right = (i < odd_n) ? i : odd_n - 1;
            even[i] -= DELTA * (odd[left] + odd[right]);
        }
    }

    /* Inverse predict 2 */
    for (int i = 0; i < odd_n - 1; i++) {
        odd[i] -= GAMMA * (even[i] + even[i + 1]);
    }
    if (odd_n > 0) {
        odd[odd_n - 1] -= GAMMA * (even[odd_n - 1] + even[half > odd_n ? odd_n : odd_n - 1]);
    }

    /* Inverse update 1 */
    if (odd_n > 0) {
        even[0] -= BETA * (odd[0] + odd[0]);
        for (int i = 1; i < half; i++) {
            int left = i - 1;
            int right = (i < odd_n) ? i : odd_n - 1;
            even[i] -= BETA * (odd[left] + odd[right]);
        }
    }

    /* Inverse predict 1 */
    for (int i = 0; i < odd_n - 1; i++) {
        odd[i] -= ALPHA * (even[i] + even[i + 1]);
    }
    if (odd_n > 0) {
        odd[odd_n - 1] -= ALPHA * (even[odd_n - 1] + even[half > odd_n ? odd_n : odd_n - 1]);
    }

    /* Interleave back - stride-1 reads */
    for (int i = 0; i < odd_n; i++) {
        data[2 * i] = even[i];
        data[2 * i + 1] = odd[i];
    }
    if (half > odd_n) {
        data[2 * odd_n] = even[odd_n];
    }
}

/*============================================================================
 * 2D Separable Transform with uniform subband sizes
 *===========================================================================*/

static void cdf97_2d_forward(float *img, int w, int h,
                             float *ll, float *lh, float *hl, float *hh,
                             int sub_w, int sub_h, float *temp) {
    /* Transform rows in-place */
    for (int y = 0; y < h; y++) {
        cdf97_fwd_1d(img + y * w, w, temp);
    }

    /* Transform columns - need to gather/scatter */
    float *col = mem_alloc(h * sizeof(float));
    if (!col) return;

    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++)
            col[y] = img[y * w + x];
        cdf97_fwd_1d(col, h, temp);
        for (int y = 0; y < h; y++)
            img[y * w + x] = col[y];
    }
    mem_free(col);

    /* Extract subbands - after 2D transform:
     * [LL | HL]   top-left is LL (low freq both dims)
     * [LH | HH]   top-right is HL (high freq in X)
     *             bottom-left is LH (high freq in Y)
     *             bottom-right is HH (high freq both)
     */
    int hw = (w + 1) / 2;  /* lowpass width */
    int hh_dim = (h + 1) / 2;  /* lowpass height */

    for (int y = 0; y < sub_h; y++) {
        for (int x = 0; x < sub_w; x++) {
            ll[y * sub_w + x] = img[y * w + x];
            if (x < w - hw && y < sub_h)
                hl[y * sub_w + x] = img[y * w + hw + x];
            if (y < h - hh_dim && x < sub_w)
                lh[y * sub_w + x] = img[(hh_dim + y) * w + x];
            if (x < w - hw && y < h - hh_dim)
                hh[y * sub_w + x] = img[(hh_dim + y) * w + hw + x];
        }
    }
}

static void cdf97_2d_inverse(const float *ll, const float *lh,
                             const float *hl, const float *hh,
                             int sub_w, int sub_h,
                             float *img, int w, int h, float *temp) {
    int hw = (w + 1) / 2;
    int hh_dim = (h + 1) / 2;

    /* Pack subbands back into image layout */
    memset(img, 0, w * h * sizeof(float));

    for (int y = 0; y < sub_h && y < hh_dim; y++) {
        for (int x = 0; x < sub_w && x < hw; x++) {
            img[y * w + x] = ll[y * sub_w + x];
        }
    }
    for (int y = 0; y < sub_h && y < hh_dim; y++) {
        for (int x = 0; x < sub_w && x < w - hw; x++) {
            img[y * w + hw + x] = hl[y * sub_w + x];
        }
    }
    for (int y = 0; y < sub_h && y < h - hh_dim; y++) {
        for (int x = 0; x < sub_w && x < hw; x++) {
            img[(hh_dim + y) * w + x] = lh[y * sub_w + x];
        }
    }
    for (int y = 0; y < sub_h && y < h - hh_dim; y++) {
        for (int x = 0; x < sub_w && x < w - hw; x++) {
            img[(hh_dim + y) * w + hw + x] = hh[y * sub_w + x];
        }
    }

    /* Inverse transform columns */
    float *col = mem_alloc(h * sizeof(float));
    if (!col) return;

    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++)
            col[y] = img[y * w + x];
        cdf97_inv_1d(col, h, temp);
        for (int y = 0; y < h; y++)
            img[y * w + x] = col[y];
    }
    mem_free(col);

    /* Inverse transform rows */
    for (int y = 0; y < h; y++) {
        cdf97_inv_1d(img + y * w, w, temp);
    }
}

/*============================================================================
 * Public API
 *===========================================================================*/

int dwt_compute_levels(int width, int height) {
    int min_dim = (width < height) ? width : height;
    int levels = 0;
    while (min_dim >= 16 && levels < WEMA_MAX_LEVELS) {
        min_dim /= 2;
        levels++;
    }
    return levels > 0 ? levels : 1;
}

int dwt_init(DWTCoeffs *coeffs, int width, int height, int num_levels) {
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
            coeffs->subbands[lev][o].coeffs = mem_calloc(w * h, sizeof(float));
            if (!coeffs->subbands[lev][o].coeffs) {
                dwt_free(coeffs);
                return -1;
            }
        }
    }

    coeffs->lowpass_w = w;
    coeffs->lowpass_h = h;
    coeffs->lowpass = mem_calloc(w * h, sizeof(float));
    if (!coeffs->lowpass) {
        dwt_free(coeffs);
        return -1;
    }

    return 0;
}

void dwt_free(DWTCoeffs *coeffs) {
    for (int lev = 0; lev < WEMA_MAX_LEVELS; lev++) {
        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            mem_free(coeffs->subbands[lev][o].coeffs);
            coeffs->subbands[lev][o].coeffs = NULL;
        }
    }
    mem_free(coeffs->lowpass);
    coeffs->lowpass = NULL;
}

void dwt_forward(const float *image, int width, int height,
                 DWTCoeffs *coeffs) {
    size_t max_size = (size_t)width * height;
    float *work = mem_alloc(max_size * sizeof(float));
    float *ll = mem_alloc(max_size * sizeof(float));
    float *lh = mem_alloc(max_size * sizeof(float));
    float *hl = mem_alloc(max_size * sizeof(float));
    float *hh = mem_alloc(max_size * sizeof(float));
    int max_dim = (width > height) ? width : height;
    float *temp = mem_alloc(max_dim * sizeof(float));

    if (!work || !ll || !lh || !hl || !hh || !temp) {
        mem_free(work);
        mem_free(ll);
        mem_free(lh);
        mem_free(hl);
        mem_free(hh);
        mem_free(temp);
        return;
    }

    memcpy(work, image, width * height * sizeof(float));

    int cur_w = width;
    int cur_h = height;

    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        int sub_w = (cur_w + 1) / 2;
        int sub_h = (cur_h + 1) / 2;

        cdf97_2d_forward(work, cur_w, cur_h, ll, lh, hl, hh, sub_w, sub_h, temp);

        /* Store subbands (with mirroring for 6 slots) */
        int n = sub_w * sub_h;
        for (int i = 0; i < n; i++) {
            coeffs->subbands[lev][0].coeffs[i] = lh[i];
            coeffs->subbands[lev][1].coeffs[i] = hh[i];
            coeffs->subbands[lev][2].coeffs[i] = hl[i];
            /* Mirror for 6 orientations (compatibility) */
            coeffs->subbands[lev][3].coeffs[i] = hl[i];
            coeffs->subbands[lev][4].coeffs[i] = hh[i];
            coeffs->subbands[lev][5].coeffs[i] = lh[i];
        }

        /* Next level operates on LL */
        memcpy(work, ll, sub_w * sub_h * sizeof(float));
        cur_w = sub_w;
        cur_h = sub_h;
    }

    memcpy(coeffs->lowpass, work, cur_w * cur_h * sizeof(float));

    mem_free(work);
    mem_free(ll);
    mem_free(lh);
    mem_free(hl);
    mem_free(hh);
    mem_free(temp);
}

void dwt_inverse(const DWTCoeffs *coeffs, float *image) {
    /* Precompute level dimensions */
    int level_w[WEMA_MAX_LEVELS + 1];
    int level_h[WEMA_MAX_LEVELS + 1];
    level_w[0] = coeffs->orig_width;
    level_h[0] = coeffs->orig_height;
    for (int lev = 1; lev <= coeffs->num_levels; lev++) {
        level_w[lev] = (level_w[lev - 1] + 1) / 2;
        level_h[lev] = (level_h[lev - 1] + 1) / 2;
    }

    size_t max_size = (size_t)coeffs->orig_width * coeffs->orig_height;
    float *work = mem_alloc(max_size * sizeof(float));
    float *output = mem_alloc(max_size * sizeof(float));
    float *lh = mem_alloc(max_size * sizeof(float));
    float *hl = mem_alloc(max_size * sizeof(float));
    float *hh = mem_alloc(max_size * sizeof(float));
    int max_dim = (coeffs->orig_width > coeffs->orig_height) ?
                   coeffs->orig_width : coeffs->orig_height;
    float *temp = mem_alloc(max_dim * sizeof(float));

    if (!work || !output || !lh || !hl || !hh || !temp) {
        mem_free(work);
        mem_free(output);
        mem_free(lh);
        mem_free(hl);
        mem_free(hh);
        mem_free(temp);
        return;
    }

    /* Start with coarsest lowpass */
    memcpy(work, coeffs->lowpass, coeffs->lowpass_w * coeffs->lowpass_h * sizeof(float));

    /* Reconstruct from coarsest to finest */
    for (int lev = coeffs->num_levels - 1; lev >= 0; lev--) {
        int out_w = level_w[lev];
        int out_h = level_h[lev];
        int sub_w = level_w[lev + 1];
        int sub_h = level_h[lev + 1];

        /* Extract subbands */
        int n = sub_w * sub_h;
        for (int i = 0; i < n; i++) {
            lh[i] = coeffs->subbands[lev][0].coeffs[i];
            hh[i] = coeffs->subbands[lev][1].coeffs[i];
            hl[i] = coeffs->subbands[lev][2].coeffs[i];
        }

        cdf97_2d_inverse(work, lh, hl, hh, sub_w, sub_h, output, out_w, out_h, temp);

        memcpy(work, output, out_w * out_h * sizeof(float));
    }

    memcpy(image, work, coeffs->orig_width * coeffs->orig_height * sizeof(float));

    mem_free(work);
    mem_free(output);
    mem_free(lh);
    mem_free(hl);
    mem_free(hh);
    mem_free(temp);
}

size_t dwt_num_positions(const DWTCoeffs *coeffs) {
    size_t total = 0;
    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            total += (size_t)coeffs->subbands[lev][o].width *
                     coeffs->subbands[lev][o].height;
        }
    }
    return total;
}
