/*
 * WEMA - Main Processing Pipeline
 *
 * Orchestrates the motion amplification process:
 * 1. RGB to grayscale
 * 2. 2D DWT decomposition
 * 3. Temporal coefficient filtering
 * 4. Coefficient amplification
 * 5. 2D DWT reconstruction
 * 6. Grayscale to RGB (blend with original)
 */

#include "wema.h"
#include "dwt.h"
#include "temporal_filter.h"
#include "phase_amp.h"
#include "alloc.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*============================================================================
 * Frame utilities
 *===========================================================================*/

int frame_alloc(Frame *frame, int width, int height, int channels) {
    frame->width = width;
    frame->height = height;
    frame->channels = channels;
    frame->data = mem_calloc((size_t)width * height * channels, sizeof(float));
    return frame->data ? 0 : -1;
}

void frame_free(Frame *frame) {
    mem_free(frame->data);
    frame->data = NULL;
}

void frame_rgb_to_gray(const Frame *rgb, float *gray) {
    int n = rgb->width * rgb->height;
    const float *src = rgb->data;

    for (int i = 0; i < n; i++) {
        /* ITU-R BT.601 luma coefficients */
        float r = src[i * 3 + 0];
        float g = src[i * 3 + 1];
        float b = src[i * 3 + 2];
        gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

/*============================================================================
 * Guided Filter Implementation
 *
 * Edge-preserving filter that uses a guide image (original luminance)
 * to preserve edges while smoothing the delta signal.
 *
 * Formula: output = a * guide + b, where a and b are computed locally
 * to minimize |output - input|^2 + epsilon * a^2
 *===========================================================================*/

static void box_filter(const float *input, float *output, int w, int h, int r) {
    /* Separable box filter with radius r */
    float *temp = mem_alloc((size_t)w * h * sizeof(float));
    if (!temp) {
        memcpy(output, input, (size_t)w * h * sizeof(float));
        return;
    }

    /* Horizontal pass */
    for (int y = 0; y < h; y++) {
        float sum = 0.0f;
        int count = 0;

        /* Initialize window */
        for (int x = 0; x <= r && x < w; x++) {
            sum += input[y * w + x];
            count++;
        }

        for (int x = 0; x < w; x++) {
            temp[y * w + x] = sum / (float)count;

            /* Slide window */
            int add_x = x + r + 1;
            int rem_x = x - r;
            if (add_x < w) {
                sum += input[y * w + add_x];
                count++;
            }
            if (rem_x >= 0) {
                sum -= input[y * w + rem_x];
                count--;
            }
        }
    }

    /* Vertical pass */
    for (int x = 0; x < w; x++) {
        float sum = 0.0f;
        int count = 0;

        /* Initialize window */
        for (int y = 0; y <= r && y < h; y++) {
            sum += temp[y * w + x];
            count++;
        }

        for (int y = 0; y < h; y++) {
            output[y * w + x] = sum / (float)count;

            /* Slide window */
            int add_y = y + r + 1;
            int rem_y = y - r;
            if (add_y < h) {
                sum += temp[add_y * w + x];
                count++;
            }
            if (rem_y >= 0) {
                sum -= temp[rem_y * w + x];
                count--;
            }
        }
    }

    mem_free(temp);
}

static void guided_filter(const float *guide, const float *input,
                          float *output, int w, int h,
                          int radius, float epsilon,
                          float *work1, float *work2, float *work3, float *work4) {
    int n = w * h;

    /* mean_I = boxfilter(guide) */
    float *mean_I = work1;
    box_filter(guide, mean_I, w, h, radius);

    /* mean_p = boxfilter(input) */
    float *mean_p = work2;
    box_filter(input, mean_p, w, h, radius);

    /* corr_I = boxfilter(guide * guide) */
    float *II = work3;
    for (int i = 0; i < n; i++) {
        II[i] = guide[i] * guide[i];
    }
    float *corr_I = work3;  /* Reuse buffer */
    box_filter(II, corr_I, w, h, radius);

    /* corr_Ip = boxfilter(guide * input) */
    float *Ip = work4;
    for (int i = 0; i < n; i++) {
        Ip[i] = guide[i] * input[i];
    }
    float *corr_Ip = work4;  /* Reuse buffer */
    box_filter(Ip, corr_Ip, w, h, radius);

    /* var_I = corr_I - mean_I * mean_I */
    /* cov_Ip = corr_Ip - mean_I * mean_p */
    /* a = cov_Ip / (var_I + epsilon) */
    /* b = mean_p - a * mean_I */
    float *a = work3;  /* Reuse corr_I buffer */
    float *b = work4;  /* Reuse corr_Ip buffer */

    for (int i = 0; i < n; i++) {
        float var_I = corr_I[i] - mean_I[i] * mean_I[i];
        float cov_Ip = corr_Ip[i] - mean_I[i] * mean_p[i];
        a[i] = cov_Ip / (var_I + epsilon);
        b[i] = mean_p[i] - a[i] * mean_I[i];
    }

    /* mean_a = boxfilter(a) */
    float *mean_a = work1;  /* Reuse mean_I buffer */
    box_filter(a, mean_a, w, h, radius);

    /* mean_b = boxfilter(b) */
    float *mean_b = work2;  /* Reuse mean_p buffer */
    box_filter(b, mean_b, w, h, radius);

    /* output = mean_a * guide + mean_b */
    for (int i = 0; i < n; i++) {
        output[i] = mean_a[i] * guide[i] + mean_b[i];
    }
}

/*============================================================================
 * Gaussian blur fallback (when edge-aware is disabled)
 *===========================================================================*/

static void gaussian_blur_5x5(const float *input, float *output, int w, int h) {
    static const float gauss5[5][5] = {
        {1, 4, 6, 4, 1},
        {4, 16, 24, 16, 4},
        {6, 24, 36, 24, 6},
        {4, 16, 24, 16, 4},
        {1, 4, 6, 4, 1}
    };

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.0f;
            float weight = 0.0f;

            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int nx = x + kx;
                    int ny = y + ky;
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                        float k = gauss5[ky + 2][kx + 2];
                        sum += k * input[ny * w + nx];
                        weight += k;
                    }
                }
            }
            output[y * w + x] = sum / weight;
        }
    }
}

/*============================================================================
 * Frame blending with optional edge-aware filtering
 *===========================================================================*/

void frame_gray_to_rgb_ex(const float *gray, const Frame *orig, Frame *out,
                          float *delta_buf, float *smooth_buf,
                          float *guide_buf, float *work1, float *work2,
                          float *work3, float *work4,
                          bool edge_aware) {
    int w = orig->width;
    int h = orig->height;
    int n = w * h;
    const float *src = orig->data;
    float *dst = out->data;

    /* Step 1: Compute original luminance (guide image) and delta */
    for (int i = 0; i < n; i++) {
        float orig_y = 0.299f * src[i * 3 + 0] +
                       0.587f * src[i * 3 + 1] +
                       0.114f * src[i * 3 + 2];
        guide_buf[i] = orig_y;
        delta_buf[i] = gray[i] - orig_y;
    }

    /* Step 2: Smooth the delta */
    if (edge_aware && work1 && work2 && work3 && work4) {
        /* Edge-aware guided filter:
         * - radius=4 gives good smoothing
         * - epsilon=0.01 controls edge sensitivity (lower = more edge-preserving) */
        guided_filter(guide_buf, delta_buf, smooth_buf, w, h,
                      4, 0.01f, work1, work2, work3, work4);
    } else {
        /* Fallback: simple Gaussian blur */
        gaussian_blur_5x5(delta_buf, smooth_buf, w, h);
    }

    /* Step 3: Add smoothed delta to original RGB */
    for (int i = 0; i < n; i++) {
        float dy = smooth_buf[i];

        dst[i * 3 + 0] = src[i * 3 + 0] + dy;
        dst[i * 3 + 1] = src[i * 3 + 1] + dy;
        dst[i * 3 + 2] = src[i * 3 + 2] + dy;

        /* Clamp to [0, 1] */
        for (int c = 0; c < 3; c++) {
            if (dst[i * 3 + c] < 0.0f) dst[i * 3 + c] = 0.0f;
            if (dst[i * 3 + c] > 1.0f) dst[i * 3 + c] = 1.0f;
        }
    }
}

/* Legacy wrapper for compatibility */
void frame_gray_to_rgb(const float *gray, const Frame *orig, Frame *out,
                       float *delta_buf, float *smooth_buf) {
    /* Allocate temporary guide buffer */
    int n = orig->width * orig->height;
    float *guide_buf = mem_alloc(n * sizeof(float));

    if (guide_buf) {
        /* Use simple Gaussian blur (no edge-aware) */
        frame_gray_to_rgb_ex(gray, orig, out, delta_buf, smooth_buf,
                             guide_buf, NULL, NULL, NULL, NULL, false);
        mem_free(guide_buf);
    } else {
        /* Fallback: direct copy */
        memcpy(out->data, orig->data, (size_t)n * 3 * sizeof(float));
    }
}

/*============================================================================
 * WEMA context management
 *===========================================================================*/

int wema_init(WemaContext *ctx, int width, int height, float fps,
              float amp_factor, float f_low, float f_high,
              int temporal_window) {
    memset(ctx, 0, sizeof(*ctx));

    ctx->width = width;
    ctx->height = height;
    ctx->fps = fps;
    ctx->amp_factor = amp_factor;
    ctx->f_low = f_low;
    ctx->f_high = f_high;
    ctx->temporal_window = temporal_window;
    ctx->edge_aware = true;       /* Enable by default */
    ctx->bilateral_temp = true;   /* Enable by default */

    /* Compute normalized frequencies */
    ctx->f_low_norm = f_low / (fps / 2.0f);
    ctx->f_high_norm = f_high / (fps / 2.0f);

    /* Clamp to valid range */
    if (ctx->f_low_norm < 0.0f) ctx->f_low_norm = 0.0f;
    if (ctx->f_high_norm > 1.0f) ctx->f_high_norm = 1.0f;

    /* Compute number of DWT levels */
    ctx->num_levels = dwt_compute_levels(width, height);

    /* Allocate DWT coefficients */
    ctx->coeffs = mem_alloc(sizeof(DWTCoeffs));
    if (!ctx->coeffs) goto error;

    if (dwt_init(ctx->coeffs, width, height, ctx->num_levels) < 0) {
        goto error;
    }

    /* Allocate temporal buffer */
    ctx->temporal_buf = mem_alloc(sizeof(TemporalBuffer));
    if (!ctx->temporal_buf) goto error;

    if (temporal_buffer_init(ctx->temporal_buf, ctx->coeffs,
                             ctx->temporal_window) < 0) {
        goto error;
    }

    /* Allocate temporal filter */
    ctx->temporal_filt = mem_alloc(sizeof(TemporalFilter));
    if (!ctx->temporal_filt) goto error;

    if (temporal_filter_init(ctx->temporal_filt,
                             ctx->f_low_norm, ctx->f_high_norm,
                             ctx->temporal_window) < 0) {
        goto error;
    }

    /* Allocate work buffers */
    size_t pixels = (size_t)width * height;
    ctx->gray_in = mem_alloc(pixels * sizeof(float));
    ctx->gray_out = mem_alloc(pixels * sizeof(float));
    ctx->delta_phi = mem_alloc(ctx->temporal_buf->num_positions * sizeof(float));
    ctx->delta_buf = mem_alloc(pixels * sizeof(float));
    ctx->smooth_buf = mem_alloc(pixels * sizeof(float));
    ctx->guide_buf = mem_alloc(pixels * sizeof(float));

    /* Work buffers for guided filter */
    ctx->guided_work[0] = mem_alloc(pixels * sizeof(float));
    ctx->guided_work[1] = mem_alloc(pixels * sizeof(float));
    ctx->guided_work[2] = mem_alloc(pixels * sizeof(float));
    ctx->guided_work[3] = mem_alloc(pixels * sizeof(float));

    if (!ctx->gray_in || !ctx->gray_out || !ctx->delta_phi ||
        !ctx->delta_buf || !ctx->smooth_buf || !ctx->guide_buf ||
        !ctx->guided_work[0] || !ctx->guided_work[1] ||
        !ctx->guided_work[2] || !ctx->guided_work[3]) {
        goto error;
    }

    return 0;

error:
    wema_free(ctx);
    return -1;
}

void wema_free(WemaContext *ctx) {
    if (ctx->coeffs) {
        dwt_free(ctx->coeffs);
        mem_free(ctx->coeffs);
        ctx->coeffs = NULL;
    }
    if (ctx->temporal_buf) {
        temporal_buffer_free(ctx->temporal_buf);
        mem_free(ctx->temporal_buf);
        ctx->temporal_buf = NULL;
    }
    if (ctx->temporal_filt) {
        temporal_filter_free(ctx->temporal_filt);
        mem_free(ctx->temporal_filt);
        ctx->temporal_filt = NULL;
    }
    mem_free(ctx->gray_in);
    mem_free(ctx->gray_out);
    mem_free(ctx->delta_phi);
    mem_free(ctx->delta_buf);
    mem_free(ctx->smooth_buf);
    mem_free(ctx->guide_buf);
    for (int i = 0; i < 4; i++) {
        mem_free(ctx->guided_work[i]);
        ctx->guided_work[i] = NULL;
    }
    ctx->gray_in = NULL;
    ctx->gray_out = NULL;
    ctx->delta_phi = NULL;
    ctx->delta_buf = NULL;
    ctx->smooth_buf = NULL;
    ctx->guide_buf = NULL;
}

bool wema_ready(const WemaContext *ctx) {
    return temporal_buffer_ready(ctx->temporal_buf);
}

int wema_process_frame(WemaContext *ctx, const Frame *in, Frame *out) {
    /* Step 1: RGB to grayscale */
    frame_rgb_to_gray(in, ctx->gray_in);

    /* Step 2: 2D DWT forward transform */
    dwt_forward(ctx->gray_in, ctx->width, ctx->height, ctx->coeffs);

    /* Step 3: Push coefficients to temporal buffer */
    temporal_buffer_push(ctx->temporal_buf, ctx->coeffs);

    /* Check if we have enough frames for filtering */
    if (!temporal_buffer_ready(ctx->temporal_buf)) {
        /* Not ready yet - copy input to output */
        memcpy(out->data, in->data,
               (size_t)in->width * in->height * in->channels * sizeof(float));
        return 0;
    }

    /* Step 4: Temporal band-pass filtering (with optional bilateral) */
    temporal_filter_apply(ctx->temporal_filt, ctx->temporal_buf, ctx->delta_phi,
                          ctx->bilateral_temp);

    /* Step 5: Coefficient amplification */
    coeff_amplify(ctx->delta_phi, ctx->temporal_buf->amplitude,
                  ctx->amp_factor, ctx->coeffs);

    /* Step 6: 2D DWT inverse transform */
    dwt_inverse(ctx->coeffs, ctx->gray_out);

    /* Step 7: Grayscale to RGB with edge-aware smoothing */
    frame_gray_to_rgb_ex(ctx->gray_out, in, out,
                         ctx->delta_buf, ctx->smooth_buf,
                         ctx->guide_buf,
                         ctx->guided_work[0], ctx->guided_work[1],
                         ctx->guided_work[2], ctx->guided_work[3],
                         ctx->edge_aware);

    return 0;
}

int wema_flush(WemaContext *ctx, Frame *out) {
    (void)ctx;
    (void)out;
    /* For now, no special flush handling needed.
     * Could be extended to output remaining buffered frames. */
    return -1;  /* No more frames */
}
