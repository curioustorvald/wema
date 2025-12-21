/*
 * WEMA - Main Processing Pipeline
 *
 * Orchestrates the motion amplification process:
 * 1. RGB to grayscale
 * 2. DT-CWT decomposition
 * 3. Temporal phase filtering
 * 4. Phase amplification
 * 5. DT-CWT reconstruction
 * 6. Grayscale to RGB (blend with original)
 */

#include "wema.h"
#include "dtcwt.h"
#include "temporal_filter.h"
#include "phase_amp.h"
#include "complex_math.h"
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

void frame_gray_to_rgb(const float *gray, const Frame *orig, Frame *out,
                       float *delta_buf, float *smooth_buf) {
    int w = orig->width;
    int h = orig->height;
    int n = w * h;
    const float *src = orig->data;
    float *dst = out->data;

    /* Step 1: Compute raw delta (gray - original luminance) */
    for (int i = 0; i < n; i++) {
        float orig_y = 0.299f * src[i * 3 + 0] +
                       0.587f * src[i * 3 + 1] +
                       0.114f * src[i * 3 + 2];
        delta_buf[i] = gray[i] - orig_y;
    }

    /* Step 2: Spatial Gaussian blur on delta (5x5 kernel) */
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
                        sum += k * delta_buf[ny * w + nx];
                        weight += k;
                    }
                }
            }
            smooth_buf[y * w + x] = sum / weight;
        }
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

/*============================================================================
 * WEMA context management
 *===========================================================================*/

int wema_init(WemaContext *ctx, int width, int height, float fps,
              float amp_factor, float f_low, float f_high) {
    memset(ctx, 0, sizeof(*ctx));

    ctx->width = width;
    ctx->height = height;
    ctx->fps = fps;
    ctx->amp_factor = amp_factor;
    ctx->f_low = f_low;
    ctx->f_high = f_high;

    /* Compute normalized frequencies */
    ctx->f_low_norm = f_low / (fps / 2.0f);
    ctx->f_high_norm = f_high / (fps / 2.0f);

    /* Clamp to valid range */
    if (ctx->f_low_norm < 0.0f) ctx->f_low_norm = 0.0f;
    if (ctx->f_high_norm > 1.0f) ctx->f_high_norm = 1.0f;

    /* Compute number of DT-CWT levels */
    ctx->num_levels = dtcwt_compute_levels(width, height);

    /* Allocate DT-CWT coefficients */
    ctx->coeffs = mem_alloc(sizeof(DTCWTCoeffs));
    if (!ctx->coeffs) goto error;

    if (dtcwt_init(ctx->coeffs, width, height, ctx->num_levels) < 0) {
        goto error;
    }

    /* Allocate temporal buffer */
    ctx->temporal_buf = mem_alloc(sizeof(TemporalBuffer));
    if (!ctx->temporal_buf) goto error;

    if (temporal_buffer_init(ctx->temporal_buf, ctx->coeffs,
                             WEMA_TEMPORAL_WINDOW) < 0) {
        goto error;
    }

    /* Allocate temporal filter */
    ctx->temporal_filt = mem_alloc(sizeof(TemporalFilter));
    if (!ctx->temporal_filt) goto error;

    if (temporal_filter_init(ctx->temporal_filt,
                             ctx->f_low_norm, ctx->f_high_norm,
                             WEMA_TEMPORAL_WINDOW) < 0) {
        goto error;
    }

    /* Allocate work buffers */
    size_t pixels = (size_t)width * height;
    ctx->gray_in = mem_alloc(pixels * sizeof(float));
    ctx->gray_out = mem_alloc(pixels * sizeof(float));
    ctx->delta_phi = mem_alloc(ctx->temporal_buf->num_positions * sizeof(float));
    ctx->delta_buf = mem_alloc(pixels * sizeof(float));
    ctx->smooth_buf = mem_alloc(pixels * sizeof(float));

    if (!ctx->gray_in || !ctx->gray_out || !ctx->delta_phi ||
        !ctx->delta_buf || !ctx->smooth_buf) {
        goto error;
    }

    return 0;

error:
    wema_free(ctx);
    return -1;
}

void wema_free(WemaContext *ctx) {
    if (ctx->coeffs) {
        dtcwt_free(ctx->coeffs);
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
    ctx->gray_in = NULL;
    ctx->gray_out = NULL;
    ctx->delta_phi = NULL;
    ctx->delta_buf = NULL;
    ctx->smooth_buf = NULL;
}

bool wema_ready(const WemaContext *ctx) {
    return temporal_buffer_ready(ctx->temporal_buf);
}

int wema_process_frame(WemaContext *ctx, const Frame *in, Frame *out) {
    /* Step 1: RGB to grayscale */
    frame_rgb_to_gray(in, ctx->gray_in);

    /* Step 2: DT-CWT forward transform */
    dtcwt_forward(ctx->gray_in, ctx->width, ctx->height, ctx->coeffs);

    /* Step 3: Push phases to temporal buffer */
    temporal_buffer_push(ctx->temporal_buf, ctx->coeffs);

    /* Check if we have enough frames for filtering */
    if (!temporal_buffer_ready(ctx->temporal_buf)) {
        /* Not ready yet - copy input to output */
        memcpy(out->data, in->data,
               (size_t)in->width * in->height * in->channels * sizeof(float));
        return 0;
    }

    /* Step 4: Temporal band-pass filtering */
    temporal_filter_apply(ctx->temporal_filt, ctx->temporal_buf, ctx->delta_phi);

    /* Step 5: Phase amplification */
    phase_amplify(ctx->delta_phi, ctx->temporal_buf->amplitude,
                  ctx->amp_factor, ctx->coeffs);

    /* Step 6: DT-CWT inverse transform */
    dtcwt_inverse(ctx->coeffs, ctx->gray_out);

    /* Step 7: Grayscale to RGB with spatial smoothing of delta */
    frame_gray_to_rgb(ctx->gray_out, in, out, ctx->delta_buf, ctx->smooth_buf);

    return 0;
}

int wema_flush(WemaContext *ctx, Frame *out) {
    (void)ctx;
    (void)out;
    /* For now, no special flush handling needed.
     * Could be extended to output remaining buffered frames. */
    return -1;  /* No more frames */
}
