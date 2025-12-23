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
#include "iir_filter.h"
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

void frame_rgb_to_gray(const Frame *rgb, float * restrict gray) {
    const int n = rgb->width * rgb->height;
    const float * restrict src = rgb->data;

    /* Vectorizable with stride-3 gather */
    for (int i = 0; i < n; i++) {
        const int idx = i * 3;
        gray[i] = 0.299f * src[idx] + 0.587f * src[idx + 1] + 0.114f * src[idx + 2];
    }
}

/*============================================================================
 * YCbCr Color Space Conversion (BT.601)
 *
 * Used for color mode processing - amplify Cb/Cr channels for blood flow.
 * RGB values expected in [0, 1], Cb/Cr centered at 0.5.
 *===========================================================================*/

static void frame_rgb_to_ycbcr(const Frame *rgb,
                                float * restrict y,
                                float * restrict cb,
                                float * restrict cr) {
    const int n = rgb->width * rgb->height;
    const float * restrict src = rgb->data;

    /* BT.601 RGB to YCbCr (Cb/Cr centered at 0.5 for [0,1] range) */
    for (int i = 0; i < n; i++) {
        const int idx = i * 3;
        const float r = src[idx + 0];
        const float g = src[idx + 1];
        const float b = src[idx + 2];

        y[i]  =  0.299f * r + 0.587f * g + 0.114f * b;
        cb[i] = -0.169f * r - 0.331f * g + 0.500f * b + 0.5f;
        cr[i] =  0.500f * r - 0.419f * g - 0.081f * b + 0.5f;
    }
}

static void frame_ycbcr_to_rgb(const float * restrict y,
                                const float * restrict cb,
                                const float * restrict cr,
                                Frame *rgb) {
    const int n = rgb->width * rgb->height;
    float * restrict dst = rgb->data;

    /* BT.601 YCbCr to RGB */
    for (int i = 0; i < n; i++) {
        const float y_val = y[i];
        const float cb_val = cb[i] - 0.5f;  /* Re-center to [-0.5, 0.5] */
        const float cr_val = cr[i] - 0.5f;

        float r = y_val + 1.402f * cr_val;
        float g = y_val - 0.344f * cb_val - 0.714f * cr_val;
        float b = y_val + 1.772f * cb_val;

        /* Clamp to [0, 1] */
        dst[i * 3 + 0] = (r < 0.0f) ? 0.0f : ((r > 1.0f) ? 1.0f : r);
        dst[i * 3 + 1] = (g < 0.0f) ? 0.0f : ((g > 1.0f) ? 1.0f : g);
        dst[i * 3 + 2] = (b < 0.0f) ? 0.0f : ((b > 1.0f) ? 1.0f : b);
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

static void box_filter(const float * restrict input, float * restrict output,
                       int w, int h, int r) {
    /*
     * Optimized box filter using prefix sums (integral image approach).
     * O(1) per pixel after prefix computation.
     */
    const size_t size = (size_t)w * h;
    int max_dim = (w > h) ? w : h;
    float * restrict temp = mem_alloc(size * sizeof(float));
    float * restrict prefix = mem_alloc((size_t)(max_dim + 1) * sizeof(float));
    if (!temp || !prefix) {
        if (temp) mem_free(temp);
        if (prefix) mem_free(prefix);
        memcpy(output, input, size * sizeof(float));
        return;
    }

    /* Horizontal pass using prefix sums */
    for (int y = 0; y < h; y++) {
        const float * restrict row_in = input + y * w;
        float * restrict row_out = temp + y * w;

        /* Build prefix sum for this row */
        prefix[0] = 0.0f;
        for (int x = 0; x < w; x++) {
            prefix[x + 1] = prefix[x] + row_in[x];
        }

        /* Compute box filter using prefix sums */
        for (int x = 0; x < w; x++) {
            int left = (x - r > 0) ? (x - r) : 0;
            int right = (x + r + 1 < w) ? (x + r + 1) : w;
            float sum = prefix[right] - prefix[left];
            row_out[x] = sum / (float)(right - left);
        }
    }

    /* Vertical pass - transpose-friendly access pattern */
    float * restrict col_prefix = prefix;  /* Reuse buffer */
    for (int x = 0; x < w; x++) {
        /* Build prefix sum for this column */
        col_prefix[0] = 0.0f;
        for (int y = 0; y < h; y++) {
            col_prefix[y + 1] = col_prefix[y] + temp[y * w + x];
        }

        /* Compute box filter using prefix sums */
        for (int y = 0; y < h; y++) {
            int top = (y - r > 0) ? (y - r) : 0;
            int bottom = (y + r + 1 < h) ? (y + r + 1) : h;
            float sum = col_prefix[bottom] - col_prefix[top];
            output[y * w + x] = sum / (float)(bottom - top);
        }
    }

    mem_free(prefix);
    mem_free(temp);
}

static void guided_filter(const float * restrict guide, const float * restrict input,
                          float * restrict output, int w, int h,
                          int radius, float epsilon,
                          float * restrict work1, float * restrict work2,
                          float * restrict work3, float * restrict work4) {
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

static void gaussian_blur_5x5(const float * restrict input, float * restrict output,
                              int w, int h) {
    /* Separable Gaussian: [1, 4, 6, 4, 1] / 16 */
    static const float kernel[5] = {1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f};

    float * restrict temp = mem_alloc((size_t)w * h * sizeof(float));
    if (!temp) {
        memcpy(output, input, (size_t)w * h * sizeof(float));
        return;
    }

    /* Horizontal pass */
    for (int y = 0; y < h; y++) {
        const float * restrict row_in = input + y * w;
        float * restrict row_out = temp + y * w;

        for (int x = 0; x < w; x++) {
            float sum = 0.0f;
            float weight = 0.0f;
            for (int k = -2; k <= 2; k++) {
                int nx = x + k;
                if (nx >= 0 && nx < w) {
                    sum += kernel[k + 2] * row_in[nx];
                    weight += kernel[k + 2];
                }
            }
            row_out[x] = sum / weight;
        }
    }

    /* Vertical pass */
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float sum = 0.0f;
            float weight = 0.0f;
            for (int k = -2; k <= 2; k++) {
                int ny = y + k;
                if (ny >= 0 && ny < h) {
                    sum += kernel[k + 2] * temp[ny * w + x];
                    weight += kernel[k + 2];
                }
            }
            output[y * w + x] = sum / weight;
        }
    }

    mem_free(temp);
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
    ctx->color_mode = false;      /* Disabled by default, set after init */

    /* Compute normalized frequencies */
    ctx->f_low_norm = f_low / (fps / 2.0f);
    ctx->f_high_norm = f_high / (fps / 2.0f);

    /* Clamp to valid range */
    if (ctx->f_low_norm < 0.0f) ctx->f_low_norm = 0.0f;
    if (ctx->f_high_norm > 1.0f) ctx->f_high_norm = 1.0f;

    /* Compute number of DWT levels */
    ctx->num_levels = dwt_compute_levels(width, height);

    /* Allocate DWT coefficients (Y channel) */
    ctx->coeffs = mem_alloc(sizeof(DWTCoeffs));
    if (!ctx->coeffs) goto error;

    if (dwt_init(ctx->coeffs, width, height, ctx->num_levels) < 0) {
        goto error;
    }

    /* Allocate temporal buffer (Y channel) */
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

/* Initialize chrominance buffers for color mode (call after wema_init) */
int wema_init_color(WemaContext *ctx) {
    size_t pixels = (size_t)ctx->width * ctx->height;

    /* Allocate DWT coefficients for Cb and Cr */
    ctx->coeffs_cb = mem_alloc(sizeof(DWTCoeffs));
    ctx->coeffs_cr = mem_alloc(sizeof(DWTCoeffs));
    if (!ctx->coeffs_cb || !ctx->coeffs_cr) goto error;

    if (dwt_init(ctx->coeffs_cb, ctx->width, ctx->height, ctx->num_levels) < 0) {
        goto error;
    }
    if (dwt_init(ctx->coeffs_cr, ctx->width, ctx->height, ctx->num_levels) < 0) {
        goto error;
    }

    /* Allocate temporal buffers for Cb and Cr */
    ctx->temporal_buf_cb = mem_alloc(sizeof(TemporalBuffer));
    ctx->temporal_buf_cr = mem_alloc(sizeof(TemporalBuffer));
    if (!ctx->temporal_buf_cb || !ctx->temporal_buf_cr) goto error;

    if (temporal_buffer_init(ctx->temporal_buf_cb, ctx->coeffs_cb,
                             ctx->temporal_window) < 0) {
        goto error;
    }
    if (temporal_buffer_init(ctx->temporal_buf_cr, ctx->coeffs_cr,
                             ctx->temporal_window) < 0) {
        goto error;
    }

    /* Allocate chrominance work buffers */
    ctx->cb_in = mem_alloc(pixels * sizeof(float));
    ctx->cr_in = mem_alloc(pixels * sizeof(float));
    ctx->cb_out = mem_alloc(pixels * sizeof(float));
    ctx->cr_out = mem_alloc(pixels * sizeof(float));
    ctx->delta_cb = mem_alloc(ctx->temporal_buf->num_positions * sizeof(float));
    ctx->delta_cr = mem_alloc(ctx->temporal_buf->num_positions * sizeof(float));

    if (!ctx->cb_in || !ctx->cr_in || !ctx->cb_out || !ctx->cr_out ||
        !ctx->delta_cb || !ctx->delta_cr) {
        goto error;
    }

    return 0;

error:
    return -1;
}

/* Initialize IIR temporal filter (call after wema_init, replaces Haar) */
int wema_init_iir(WemaContext *ctx) {
    size_t num_positions = dwt_num_positions(ctx->coeffs);

    /* Allocate IIR filter for Y channel */
    ctx->iir_filt = mem_alloc(sizeof(IIRTemporalFilter));
    if (!ctx->iir_filt) goto error;

    if (iir_temporal_init(ctx->iir_filt, num_positions,
                          ctx->f_low, ctx->f_high, ctx->fps) < 0) {
        goto error;
    }

    /* Allocate previous coefficients buffer for computing delta */
    ctx->prev_coeffs = mem_calloc(num_positions, sizeof(float));
    if (!ctx->prev_coeffs) goto error;

    /* Free Haar temporal resources - no longer needed */
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

    ctx->use_iir = true;
    return 0;

error:
    if (ctx->iir_filt) {
        iir_temporal_free(ctx->iir_filt);
        mem_free(ctx->iir_filt);
        ctx->iir_filt = NULL;
    }
    mem_free(ctx->prev_coeffs);
    ctx->prev_coeffs = NULL;
    return -1;
}

/* Initialize IIR filters for color mode chrominance channels */
int wema_init_iir_color(WemaContext *ctx) {
    if (!ctx->coeffs_cb || !ctx->coeffs_cr) return -1;

    size_t num_positions = dwt_num_positions(ctx->coeffs_cb);

    /* Allocate IIR filters for Cb and Cr */
    ctx->iir_filt_cb = mem_alloc(sizeof(IIRTemporalFilter));
    ctx->iir_filt_cr = mem_alloc(sizeof(IIRTemporalFilter));
    if (!ctx->iir_filt_cb || !ctx->iir_filt_cr) goto error;

    if (iir_temporal_init(ctx->iir_filt_cb, num_positions,
                          ctx->f_low, ctx->f_high, ctx->fps) < 0) {
        goto error;
    }
    if (iir_temporal_init(ctx->iir_filt_cr, num_positions,
                          ctx->f_low, ctx->f_high, ctx->fps) < 0) {
        goto error;
    }

    /* Allocate previous coefficient buffers */
    ctx->prev_coeffs_cb = mem_calloc(num_positions, sizeof(float));
    ctx->prev_coeffs_cr = mem_calloc(num_positions, sizeof(float));
    if (!ctx->prev_coeffs_cb || !ctx->prev_coeffs_cr) goto error;

    /* Free Haar temporal resources for chrominance */
    if (ctx->temporal_buf_cb) {
        temporal_buffer_free(ctx->temporal_buf_cb);
        mem_free(ctx->temporal_buf_cb);
        ctx->temporal_buf_cb = NULL;
    }
    if (ctx->temporal_buf_cr) {
        temporal_buffer_free(ctx->temporal_buf_cr);
        mem_free(ctx->temporal_buf_cr);
        ctx->temporal_buf_cr = NULL;
    }

    return 0;

error:
    if (ctx->iir_filt_cb) {
        iir_temporal_free(ctx->iir_filt_cb);
        mem_free(ctx->iir_filt_cb);
        ctx->iir_filt_cb = NULL;
    }
    if (ctx->iir_filt_cr) {
        iir_temporal_free(ctx->iir_filt_cr);
        mem_free(ctx->iir_filt_cr);
        ctx->iir_filt_cr = NULL;
    }
    mem_free(ctx->prev_coeffs_cb);
    mem_free(ctx->prev_coeffs_cr);
    ctx->prev_coeffs_cb = NULL;
    ctx->prev_coeffs_cr = NULL;
    return -1;
}

void wema_free(WemaContext *ctx) {
    /* Free Y channel state */
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

    /* Free chrominance state (color mode) */
    if (ctx->coeffs_cb) {
        dwt_free(ctx->coeffs_cb);
        mem_free(ctx->coeffs_cb);
        ctx->coeffs_cb = NULL;
    }
    if (ctx->coeffs_cr) {
        dwt_free(ctx->coeffs_cr);
        mem_free(ctx->coeffs_cr);
        ctx->coeffs_cr = NULL;
    }
    if (ctx->temporal_buf_cb) {
        temporal_buffer_free(ctx->temporal_buf_cb);
        mem_free(ctx->temporal_buf_cb);
        ctx->temporal_buf_cb = NULL;
    }
    if (ctx->temporal_buf_cr) {
        temporal_buffer_free(ctx->temporal_buf_cr);
        mem_free(ctx->temporal_buf_cr);
        ctx->temporal_buf_cr = NULL;
    }

    /* Free IIR filter state */
    if (ctx->iir_filt) {
        iir_temporal_free(ctx->iir_filt);
        mem_free(ctx->iir_filt);
        ctx->iir_filt = NULL;
    }
    if (ctx->iir_filt_cb) {
        iir_temporal_free(ctx->iir_filt_cb);
        mem_free(ctx->iir_filt_cb);
        ctx->iir_filt_cb = NULL;
    }
    if (ctx->iir_filt_cr) {
        iir_temporal_free(ctx->iir_filt_cr);
        mem_free(ctx->iir_filt_cr);
        ctx->iir_filt_cr = NULL;
    }
    mem_free(ctx->prev_coeffs);
    mem_free(ctx->prev_coeffs_cb);
    mem_free(ctx->prev_coeffs_cr);
    ctx->prev_coeffs = NULL;
    ctx->prev_coeffs_cb = NULL;
    ctx->prev_coeffs_cr = NULL;

    /* Free Y channel work buffers */
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

    /* Free chrominance work buffers */
    mem_free(ctx->cb_in);
    mem_free(ctx->cr_in);
    mem_free(ctx->cb_out);
    mem_free(ctx->cr_out);
    mem_free(ctx->delta_cb);
    mem_free(ctx->delta_cr);
    ctx->cb_in = NULL;
    ctx->cr_in = NULL;
    ctx->cb_out = NULL;
    ctx->cr_out = NULL;
    ctx->delta_cb = NULL;
    ctx->delta_cr = NULL;
}

bool wema_ready(const WemaContext *ctx) {
    if (ctx->use_iir) {
        return ctx->iir_filt && iir_temporal_ready(ctx->iir_filt);
    }
    return temporal_buffer_ready(ctx->temporal_buf);
}

/*============================================================================
 * IIR-mode helper: Extract DWT coefficients to flat array
 *===========================================================================*/

static void extract_coeffs_to_flat(const DWTCoeffs *coeffs, float *flat) {
    size_t pos = 0;
    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            const Subband *sub = &coeffs->subbands[lev][o];
            int n = sub->width * sub->height;
            memcpy(flat + pos, sub->coeffs, n * sizeof(float));
            pos += n;
        }
    }
}

/*============================================================================
 * IIR-mode helper: Apply amplified delta back to DWT coefficients
 *===========================================================================*/

static void apply_amplified_delta(const float *delta, const float *orig,
                                   float amp_factor, DWTCoeffs *coeffs) {
    /* Compute coefficient magnitude statistics for thresholding */
    size_t total_pos = dwt_num_positions(coeffs);

    float mag_sum = 0.0f;
    for (size_t i = 0; i < total_pos; i++) {
        mag_sum += fabsf(orig[i]);
    }
    const float coeff_threshold = mag_sum / (float)total_pos * 0.2f;

    /* Apply amplification per level */
    size_t pos = 0;
    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        const float level_alpha = amp_factor * ((lev == 0) ? 0.5f : 1.0f);

        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            Subband *sub = &coeffs->subbands[lev][o];
            float *out = sub->coeffs;
            const int n = sub->width * sub->height;

            for (int i = 0; i < n; i++) {
                const float o_val = orig[pos];
                const float d_val = delta[pos];
                const float mask = (fabsf(o_val) > coeff_threshold) ? 1.0f : 0.0f;
                out[i] = o_val + mask * level_alpha * d_val;
                pos++;
            }
        }
    }
}

int wema_process_frame(WemaContext *ctx, const Frame *in, Frame *out) {
    const size_t pixels = (size_t)ctx->width * ctx->height;

    /*========================================================================
     * IIR Mode: Fast streaming temporal filter
     *=======================================================================*/
    if (ctx->use_iir) {
        if (ctx->color_mode) {
            /* IIR Color mode: Process Y, Cb, Cr */
            frame_rgb_to_ycbcr(in, ctx->gray_in, ctx->cb_in, ctx->cr_in);

            /* DWT forward */
            dwt_forward(ctx->gray_in, ctx->width, ctx->height, ctx->coeffs);
            dwt_forward(ctx->cb_in, ctx->width, ctx->height, ctx->coeffs_cb);
            dwt_forward(ctx->cr_in, ctx->width, ctx->height, ctx->coeffs_cr);

            /* Extract coefficients to flat arrays */
            extract_coeffs_to_flat(ctx->coeffs, ctx->prev_coeffs);
            extract_coeffs_to_flat(ctx->coeffs_cb, ctx->prev_coeffs_cb);
            extract_coeffs_to_flat(ctx->coeffs_cr, ctx->prev_coeffs_cr);

            /* IIR temporal filtering */
            iir_temporal_process(ctx->iir_filt, ctx->prev_coeffs, ctx->delta_phi);
            iir_temporal_process(ctx->iir_filt_cb, ctx->prev_coeffs_cb, ctx->delta_cb);
            iir_temporal_process(ctx->iir_filt_cr, ctx->prev_coeffs_cr, ctx->delta_cr);

            /* Check warmup */
            if (!iir_temporal_ready(ctx->iir_filt)) {
                memcpy(out->data, in->data, pixels * in->channels * sizeof(float));
                return 0;
            }

            /* Apply amplification */
            apply_amplified_delta(ctx->delta_phi, ctx->prev_coeffs,
                                   ctx->amp_factor, ctx->coeffs);
            apply_amplified_delta(ctx->delta_cb, ctx->prev_coeffs_cb,
                                   ctx->amp_factor * 0.5f, ctx->coeffs_cb);
            apply_amplified_delta(ctx->delta_cr, ctx->prev_coeffs_cr,
                                   ctx->amp_factor * 0.5f, ctx->coeffs_cr);

            /* DWT inverse */
            dwt_inverse(ctx->coeffs, ctx->gray_out);
            dwt_inverse(ctx->coeffs_cb, ctx->cb_out);
            dwt_inverse(ctx->coeffs_cr, ctx->cr_out);

            /* YCbCr to RGB */
            frame_ycbcr_to_rgb(ctx->gray_out, ctx->cb_out, ctx->cr_out, out);

        } else {
            /* IIR Standard mode: Process luminance only */
            frame_rgb_to_gray(in, ctx->gray_in);

            /* DWT forward */
            dwt_forward(ctx->gray_in, ctx->width, ctx->height, ctx->coeffs);

            /* Extract coefficients to flat array */
            extract_coeffs_to_flat(ctx->coeffs, ctx->prev_coeffs);

            /* IIR temporal filtering */
            iir_temporal_process(ctx->iir_filt, ctx->prev_coeffs, ctx->delta_phi);

            /* Check warmup */
            if (!iir_temporal_ready(ctx->iir_filt)) {
                memcpy(out->data, in->data, pixels * in->channels * sizeof(float));
                return 0;
            }

            /* Apply amplification */
            apply_amplified_delta(ctx->delta_phi, ctx->prev_coeffs,
                                   ctx->amp_factor, ctx->coeffs);

            /* DWT inverse */
            dwt_inverse(ctx->coeffs, ctx->gray_out);

            /* Grayscale to RGB with edge-aware smoothing */
            frame_gray_to_rgb_ex(ctx->gray_out, in, out,
                                 ctx->delta_buf, ctx->smooth_buf,
                                 ctx->guide_buf,
                                 ctx->guided_work[0], ctx->guided_work[1],
                                 ctx->guided_work[2], ctx->guided_work[3],
                                 ctx->edge_aware);
        }
        return 0;
    }

    /*========================================================================
     * Haar Mode: Original wavelet-based temporal filter
     *=======================================================================*/
    if (ctx->color_mode) {
        /* Color mode: Process Y, Cb, Cr separately */

        /* Step 1: RGB to YCbCr */
        frame_rgb_to_ycbcr(in, ctx->gray_in, ctx->cb_in, ctx->cr_in);

        /* Step 2: 2D DWT forward transform for all channels */
        dwt_forward(ctx->gray_in, ctx->width, ctx->height, ctx->coeffs);
        dwt_forward(ctx->cb_in, ctx->width, ctx->height, ctx->coeffs_cb);
        dwt_forward(ctx->cr_in, ctx->width, ctx->height, ctx->coeffs_cr);

        /* Step 3: Push coefficients to temporal buffers */
        temporal_buffer_push(ctx->temporal_buf, ctx->coeffs);
        temporal_buffer_push(ctx->temporal_buf_cb, ctx->coeffs_cb);
        temporal_buffer_push(ctx->temporal_buf_cr, ctx->coeffs_cr);

        /* Check if we have enough frames for filtering */
        if (!temporal_buffer_ready(ctx->temporal_buf)) {
            /* Not ready yet - copy input to output */
            memcpy(out->data, in->data,
                   pixels * in->channels * sizeof(float));
            return 0;
        }

        /* Step 4: Temporal band-pass filtering (with optional bilateral) */
        temporal_filter_apply(ctx->temporal_filt, ctx->temporal_buf,
                              ctx->delta_phi, ctx->bilateral_temp);
        temporal_filter_apply(ctx->temporal_filt, ctx->temporal_buf_cb,
                              ctx->delta_cb, ctx->bilateral_temp);
        temporal_filter_apply(ctx->temporal_filt, ctx->temporal_buf_cr,
                              ctx->delta_cr, ctx->bilateral_temp);

        /* Step 5: Coefficient amplification for all channels */
        coeff_amplify(ctx->delta_phi, ctx->temporal_buf->amplitude,
                      ctx->amp_factor, ctx->coeffs);
        /* Chrominance uses reduced amplification (blood flow subtle) */
        coeff_amplify(ctx->delta_cb, ctx->temporal_buf_cb->amplitude,
                      ctx->amp_factor * 0.5f, ctx->coeffs_cb);
        coeff_amplify(ctx->delta_cr, ctx->temporal_buf_cr->amplitude,
                      ctx->amp_factor * 0.5f, ctx->coeffs_cr);

        /* Step 6: 2D DWT inverse transform for all channels */
        dwt_inverse(ctx->coeffs, ctx->gray_out);
        dwt_inverse(ctx->coeffs_cb, ctx->cb_out);
        dwt_inverse(ctx->coeffs_cr, ctx->cr_out);

        /* Step 7: YCbCr to RGB */
        frame_ycbcr_to_rgb(ctx->gray_out, ctx->cb_out, ctx->cr_out, out);

    } else {
        /* Standard mode: Process luminance only */

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
                   pixels * in->channels * sizeof(float));
            return 0;
        }

        /* Step 4: Temporal band-pass filtering (with optional bilateral) */
        temporal_filter_apply(ctx->temporal_filt, ctx->temporal_buf,
                              ctx->delta_phi, ctx->bilateral_temp);

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
    }

    return 0;
}

int wema_flush(WemaContext *ctx, Frame *out) {
    (void)ctx;
    (void)out;
    /* For now, no special flush handling needed.
     * Could be extended to output remaining buffered frames. */
    return -1;  /* No more frames */
}
