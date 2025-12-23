/*
 * WEMA Batch Processing - Fully Parallel Pipeline
 *
 * Processes frames in batches with OpenMP parallelization:
 * - Phase 1: Spatial DWT (parallel across frames)
 * - Phase 2: Temporal IIR filtering (parallel across positions)
 * - Phase 3: Spatial IDWT + edge-aware smoothing (parallel across frames)
 */

#include "wema.h"
#include "dwt.h"
#include "iir_filter.h"
#include "alloc.h"

#include <string.h>
#include <math.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/*============================================================================
 * Box filter (for guided filter) - thread-safe version with pre-allocated temp
 *===========================================================================*/

static void box_filter_inplace(const float *input, float *output,
                                int w, int h, int r,
                                float *temp, float *prefix) {
    /* Horizontal pass using prefix sums */
    for (int y = 0; y < h; y++) {
        const float * restrict row_in = input + y * w;
        float * restrict row_out = temp + y * w;

        prefix[0] = 0.0f;
        for (int x = 0; x < w; x++) {
            prefix[x + 1] = prefix[x] + row_in[x];
        }

        for (int x = 0; x < w; x++) {
            int left = (x - r > 0) ? (x - r) : 0;
            int right = (x + r + 1 < w) ? (x + r + 1) : w;
            float sum = prefix[right] - prefix[left];
            row_out[x] = sum / (float)(right - left);
        }
    }

    /* Vertical pass */
    for (int x = 0; x < w; x++) {
        prefix[0] = 0.0f;
        for (int y = 0; y < h; y++) {
            prefix[y + 1] = prefix[y] + temp[y * w + x];
        }

        for (int y = 0; y < h; y++) {
            int top = (y - r > 0) ? (y - r) : 0;
            int bottom = (y + r + 1 < h) ? (y + r + 1) : h;
            float sum = prefix[bottom] - prefix[top];
            output[y * w + x] = sum / (float)(bottom - top);
        }
    }
}

/*============================================================================
 * Guided filter - edge-aware smoothing (thread-safe)
 *
 * work buffers layout (9 buffers of size w*h each):
 *   [0] = mean_I / mean_a
 *   [1] = mean_p / mean_b
 *   [2] = corr_I / a (II product, then a coefficients)
 *   [3] = corr_Ip / b (Ip product, then b coefficients)
 *   [4] = temp for box filter
 *   [5] = prefix for box filter
 *   [6] = guide image (original luminance)
 *   [7] = delta input (gray_out - orig_y)
 *   [8] = smooth output (filtered delta)
 *===========================================================================*/

static void guided_filter_batch(const float * restrict guide,
                                 const float * restrict input,
                                 float * restrict output,
                                 int w, int h, int radius, float epsilon,
                                 float * restrict work) {
    const int n = w * h;

    float *mean_I = work;
    float *mean_p = work + n;
    float *corr_I = work + 2 * n;
    float *corr_Ip = work + 3 * n;
    float *temp = work + 4 * n;
    float *prefix = work + 5 * n;

    /* mean_I = boxfilter(guide) */
    box_filter_inplace(guide, mean_I, w, h, radius, temp, prefix);

    /* mean_p = boxfilter(input) */
    box_filter_inplace(input, mean_p, w, h, radius, temp, prefix);

    /* corr_I = boxfilter(guide * guide) */
    for (int i = 0; i < n; i++) {
        corr_I[i] = guide[i] * guide[i];
    }
    box_filter_inplace(corr_I, corr_I, w, h, radius, temp, prefix);

    /* corr_Ip = boxfilter(guide * input) */
    for (int i = 0; i < n; i++) {
        corr_Ip[i] = guide[i] * input[i];
    }
    box_filter_inplace(corr_Ip, corr_Ip, w, h, radius, temp, prefix);

    /* Compute a and b coefficients (reuse corr_I and corr_Ip buffers) */
    float *a = corr_I;
    float *b = corr_Ip;
    for (int i = 0; i < n; i++) {
        float var_I = corr_I[i] - mean_I[i] * mean_I[i];
        float cov_Ip = corr_Ip[i] - mean_I[i] * mean_p[i];
        a[i] = cov_Ip / (var_I + epsilon);
        b[i] = mean_p[i] - a[i] * mean_I[i];
    }

    /* mean_a = boxfilter(a), reuse mean_I buffer */
    float *mean_a = mean_I;
    box_filter_inplace(a, mean_a, w, h, radius, temp, prefix);

    /* mean_b = boxfilter(b), reuse mean_p buffer */
    float *mean_b = mean_p;
    box_filter_inplace(b, mean_b, w, h, radius, temp, prefix);

    /* output = mean_a * guide + mean_b */
    for (int i = 0; i < n; i++) {
        output[i] = mean_a[i] * guide[i] + mean_b[i];
    }
}

/*============================================================================
 * Batch context initialization
 *===========================================================================*/

int wema_batch_init(WemaBatchContext *batch, const WemaContext *wema,
                    int batch_size, int num_threads) {
    memset(batch, 0, sizeof(*batch));

    if (batch_size < 1) batch_size = WEMA_BATCH_SIZE_DEF;
    if (batch_size > WEMA_BATCH_SIZE_MAX) batch_size = WEMA_BATCH_SIZE_MAX;

    batch->batch_size = batch_size;
    batch->width = wema->width;
    batch->height = wema->height;

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
        batch->num_threads = num_threads;
    } else {
        batch->num_threads = omp_get_max_threads();
    }
#else
    batch->num_threads = 1;
#endif

    /* Get number of DWT coefficient positions */
    batch->num_positions = dwt_num_positions(wema->coeffs);

    const size_t pixels = (size_t)wema->width * wema->height;

    /* Allocate batch buffers for coefficients */
    const size_t coeff_batch_size = (size_t)batch_size * batch->num_positions;
    batch->coeffs_batch = mem_alloc(coeff_batch_size * sizeof(float));
    batch->delta_batch = mem_alloc(coeff_batch_size * sizeof(float));

    if (!batch->coeffs_batch || !batch->delta_batch) {
        wema_batch_free(batch);
        return -1;
    }

    /* Allocate batch frame buffers */
    const size_t gray_batch_size = (size_t)batch_size * pixels;
    batch->gray_batch = mem_alloc(gray_batch_size * sizeof(float));
    batch->out_batch = mem_alloc(gray_batch_size * sizeof(float));

    if (!batch->gray_batch || !batch->out_batch) {
        wema_batch_free(batch);
        return -1;
    }

    /* Allocate per-frame DWT structures for parallel Phase 1 & 3 */
    batch->dwt_batch = mem_alloc((size_t)batch_size * sizeof(DWTCoeffs));
    if (!batch->dwt_batch) {
        wema_batch_free(batch);
        return -1;
    }

    for (int i = 0; i < batch_size; i++) {
        if (dwt_init(&batch->dwt_batch[i], wema->width, wema->height,
                     wema->num_levels) < 0) {
            /* Cleanup already initialized */
            for (int j = 0; j < i; j++) {
                dwt_free(&batch->dwt_batch[j]);
            }
            wema_batch_free(batch);
            return -1;
        }
    }

    /* Allocate per-thread smoothing buffers for edge-aware filter
     * Each thread needs 9 buffers of size pixels:
     * [0-3] guided filter intermediates, [4] temp, [5] prefix,
     * [6] guide, [7] delta, [8] smooth */
    batch->smooth_buf_stride = 9 * (int)pixels;
    batch->smooth_buffers = mem_alloc((size_t)batch->num_threads *
                                       batch->smooth_buf_stride * sizeof(float));
    if (!batch->smooth_buffers) {
        wema_batch_free(batch);
        return -1;
    }

    return 0;
}

void wema_batch_free(WemaBatchContext *batch) {
    mem_free(batch->coeffs_batch);
    mem_free(batch->delta_batch);
    mem_free(batch->gray_batch);
    mem_free(batch->out_batch);
    mem_free(batch->smooth_buffers);

    if (batch->dwt_batch) {
        for (int i = 0; i < batch->batch_size; i++) {
            dwt_free(&batch->dwt_batch[i]);
        }
        mem_free(batch->dwt_batch);
    }

    memset(batch, 0, sizeof(*batch));
}

/*============================================================================
 * Helper: RGB to grayscale
 *===========================================================================*/

static inline void rgb_to_gray(const float *rgb, float *gray, size_t pixels) {
    for (size_t i = 0; i < pixels; i++) {
        gray[i] = 0.299f * rgb[i * 3 + 0] +
                  0.587f * rgb[i * 3 + 1] +
                  0.114f * rgb[i * 3 + 2];
    }
}

/*============================================================================
 * Helper: Extract DWT coefficients to flat array
 *===========================================================================*/

static void extract_coeffs_flat(const DWTCoeffs *coeffs, float *flat) {
    size_t pos = 0;
    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            const Subband *sub = &coeffs->subbands[lev][o];
            const size_t n = (size_t)sub->width * sub->height;
            memcpy(flat + pos, sub->coeffs, n * sizeof(float));
            pos += n;
        }
    }
}

/*============================================================================
 * Helper: Apply amplified delta back to DWT coefficients
 *===========================================================================*/

static void apply_delta_to_coeffs(const float *delta, const float *orig,
                                   float amp_factor, DWTCoeffs *coeffs,
                                   size_t num_positions) {
    /* Compute coefficient magnitude threshold */
    float mag_sum = 0.0f;
    for (size_t i = 0; i < num_positions; i++) {
        mag_sum += fabsf(orig[i]);
    }
    const float coeff_threshold = mag_sum / (float)num_positions * 0.2f;

    /* Apply amplification per level */
    size_t pos = 0;
    for (int lev = 0; lev < coeffs->num_levels; lev++) {
        const float level_alpha = amp_factor * ((lev == 0) ? 0.5f : 1.0f);

        for (int o = 0; o < WEMA_NUM_ORIENTATIONS; o++) {
            Subband *sub = &coeffs->subbands[lev][o];
            float *out = sub->coeffs;
            const size_t n = (size_t)sub->width * sub->height;

            for (size_t i = 0; i < n; i++) {
                const float o_val = orig[pos];
                const float d_val = delta[pos];
                const float mask = (fabsf(o_val) > coeff_threshold) ? 1.0f : 0.0f;
                out[i] = o_val + mask * level_alpha * d_val;
                pos++;
            }
        }
    }
}

/*============================================================================
 * Helper: Grayscale to RGB with edge-aware smoothing
 *
 * work buffer layout (9 buffers of size w*h each):
 *   [0-3] = guided filter intermediate buffers
 *   [4]   = temp for box filter
 *   [5]   = prefix for box filter
 *   [6]   = guide image (original luminance)
 *   [7]   = delta (input to guided filter)
 *   [8]   = smooth (output from guided filter)
 *===========================================================================*/

static void gray_to_rgb_edge_aware(const float *gray_out, const Frame *orig,
                                    float *out_rgb, int w, int h,
                                    float *work) {
    const size_t n = (size_t)w * h;
    const float *src = orig->data;

    float *guide = work + 6 * n;
    float *delta = work + 7 * n;
    float *smooth = work + 8 * n;

    /* Compute original luminance (guide) and raw delta */
    for (size_t i = 0; i < n; i++) {
        float orig_y = 0.299f * src[i * 3 + 0] +
                       0.587f * src[i * 3 + 1] +
                       0.114f * src[i * 3 + 2];
        guide[i] = orig_y;
        delta[i] = gray_out[i] - orig_y;
    }

    /* Apply edge-aware guided filter to delta */
    guided_filter_batch(guide, delta, smooth, w, h, 4, 0.01f, work);

    /* Add smoothed delta to original RGB */
    for (size_t i = 0; i < n; i++) {
        float dy = smooth[i];

        float r = src[i * 3 + 0] + dy;
        float g = src[i * 3 + 1] + dy;
        float b = src[i * 3 + 2] + dy;

        /* Clamp to [0, 1] */
        out_rgb[i * 3 + 0] = (r < 0.0f) ? 0.0f : ((r > 1.0f) ? 1.0f : r);
        out_rgb[i * 3 + 1] = (g < 0.0f) ? 0.0f : ((g > 1.0f) ? 1.0f : g);
        out_rgb[i * 3 + 2] = (b < 0.0f) ? 0.0f : ((b > 1.0f) ? 1.0f : b);
    }
}

/*============================================================================
 * Main batch processing function - Fully Parallel
 *===========================================================================*/

int wema_batch_process(WemaContext *wema, WemaBatchContext *batch,
                       const Frame *frames_in, Frame *frames_out,
                       int frame_count) {
    if (frame_count <= 0) return 0;
    if (frame_count > batch->batch_size) {
        frame_count = batch->batch_size;
    }

    const int w = batch->width;
    const int h = batch->height;
    const size_t pixels = (size_t)w * h;
    const size_t num_pos = batch->num_positions;

    /*========================================================================
     * PHASE 1: RGB→Gray + Spatial DWT (parallel across frames)
     *=======================================================================*/
    #pragma omp parallel for schedule(static)
    for (int f = 0; f < frame_count; f++) {
        const Frame *in = &frames_in[f];
        float *gray = batch->gray_batch + f * pixels;
        float *coeffs = batch->coeffs_batch + f * num_pos;
        DWTCoeffs *dwt = &batch->dwt_batch[f];

        /* RGB to grayscale */
        rgb_to_gray(in->data, gray, pixels);

        /* DWT forward (each frame has its own DWT context) */
        dwt_forward(gray, w, h, dwt);

        /* Extract to flat array */
        extract_coeffs_flat(dwt, coeffs);
    }

    /*========================================================================
     * PHASE 2: Temporal IIR filtering (parallel across positions)
     *=======================================================================*/
    iir_temporal_process_batch(wema->iir_filt,
                               batch->coeffs_batch,
                               batch->delta_batch,
                               frame_count);

    /* Compute valid output range based on warmup */
    const int warmup_done = wema->iir_filt->warmup_frames;
    const int frames_start = wema->iir_filt->frames_processed - frame_count;
    int valid_start = (frames_start < warmup_done) ? (warmup_done - frames_start) : 0;
    if (valid_start > frame_count) valid_start = frame_count;

    const int valid_count = frame_count - valid_start;

    /*========================================================================
     * PHASE 3: Amplification + IDWT + Edge-aware smoothing (parallel)
     *=======================================================================*/
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < valid_count; i++) {
        const int f = valid_start + i;
        const Frame *in = &frames_in[f];
        Frame *out = &frames_out[i];
        const float *coeffs = batch->coeffs_batch + f * num_pos;
        const float *delta = batch->delta_batch + f * num_pos;
        float *gray_out = batch->out_batch + f * pixels;
        DWTCoeffs *dwt = &batch->dwt_batch[f];

        /* Get per-thread work buffer for smoothing */
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        float *work = batch->smooth_buffers + tid * batch->smooth_buf_stride;

        /* Apply amplified delta to DWT coefficients */
        apply_delta_to_coeffs(delta, coeffs, wema->amp_factor, dwt, num_pos);

        /* IDWT (each frame has its own DWT context) */
        dwt_inverse(dwt, gray_out);

        /* Grayscale to RGB with edge-aware smoothing */
        gray_to_rgb_edge_aware(gray_out, in, out->data, w, h, work);
    }

    return valid_count;
}
