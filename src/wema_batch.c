/*
 * WEMA Batch Processing - Parallel multi-frame pipeline
 *
 * Processes frames in batches for better parallelism:
 * - Phase 1: Spatial DWT (parallel across frames)
 * - Phase 2: Temporal IIR filtering (parallel across positions)
 * - Phase 3: Spatial IDWT (parallel across frames)
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
 * Batch context initialization
 *===========================================================================*/

int wema_batch_init(WemaBatchContext *batch, const WemaContext *wema,
                    int batch_size, int num_threads) {
    memset(batch, 0, sizeof(*batch));

    if (batch_size < 1) batch_size = WEMA_BATCH_SIZE_DEF;
    if (batch_size > WEMA_BATCH_SIZE_MAX) batch_size = WEMA_BATCH_SIZE_MAX;

    batch->batch_size = batch_size;
    batch->num_threads = num_threads;

    /* Get number of DWT coefficient positions */
    batch->num_positions = dwt_num_positions(wema->coeffs);

    size_t pixels = (size_t)wema->width * wema->height;

    /* Allocate batch buffers for coefficients */
    size_t coeff_batch_size = (size_t)batch_size * batch->num_positions;
    batch->coeffs_batch = mem_alloc(coeff_batch_size * sizeof(float));
    batch->delta_batch = mem_alloc(coeff_batch_size * sizeof(float));
    batch->prev_coeffs = mem_calloc(batch->num_positions, sizeof(float));

    if (!batch->coeffs_batch || !batch->delta_batch || !batch->prev_coeffs) {
        wema_batch_free(batch);
        return -1;
    }

    /* Allocate batch frame buffers */
    size_t gray_batch_size = (size_t)batch_size * pixels;
    batch->gray_batch = mem_alloc(gray_batch_size * sizeof(float));
    batch->out_batch = mem_alloc(gray_batch_size * sizeof(float));

    if (!batch->gray_batch || !batch->out_batch) {
        wema_batch_free(batch);
        return -1;
    }

#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#endif

    return 0;
}

void wema_batch_free(WemaBatchContext *batch) {
    mem_free(batch->coeffs_batch);
    mem_free(batch->delta_batch);
    mem_free(batch->prev_coeffs);
    mem_free(batch->gray_batch);
    mem_free(batch->out_batch);
    memset(batch, 0, sizeof(*batch));
}

/*============================================================================
 * Helper: RGB to grayscale
 *===========================================================================*/

static void rgb_to_gray(const float *rgb, float *gray, size_t pixels) {
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
            size_t n = (size_t)sub->width * sub->height;
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
 * Helper: Grayscale output to RGB with edge-aware smoothing
 *===========================================================================*/

static void gray_to_rgb_simple(const float *gray, const Frame *orig,
                                float *out_rgb, int w, int h) {
    const size_t n = (size_t)w * h;
    const float *src = orig->data;

    for (size_t i = 0; i < n; i++) {
        /* Compute original luminance */
        float orig_y = 0.299f * src[i * 3 + 0] +
                       0.587f * src[i * 3 + 1] +
                       0.114f * src[i * 3 + 2];

        /* Compute delta and add to RGB channels */
        float dy = gray[i] - orig_y;

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
 * Main batch processing function
 *===========================================================================*/

int wema_batch_process(WemaContext *wema, WemaBatchContext *batch,
                       const Frame *frames_in, Frame *frames_out,
                       int frame_count) {
    if (frame_count <= 0) return 0;
    if (frame_count > batch->batch_size) {
        frame_count = batch->batch_size;
    }

    const int w = wema->width;
    const int h = wema->height;
    const size_t pixels = (size_t)w * h;
    const size_t num_pos = batch->num_positions;

    /* We need per-frame DWT structures for parallel processing.
     * For simplicity, we'll allocate temporary ones or process sequentially
     * with the shared coeffs. Let's use a hybrid approach:
     * - Phase 1 & 3 can share wema->coeffs if processed sequentially per frame
     * - Phase 2 is where the parallelism happens (across positions)
     *
     * For now, we'll process Phase 1 and 3 sequentially per frame
     * but parallelize Phase 2 which is the expensive temporal filtering.
     */

    /*========================================================================
     * PHASE 1: Spatial DWT for all frames (sequential - uses shared DWT state)
     * TODO: Parallelize by allocating per-thread DWT structures
     *=======================================================================*/
    for (int f = 0; f < frame_count; f++) {
        const Frame *in = &frames_in[f];
        float *gray = batch->gray_batch + f * pixels;
        float *coeffs = batch->coeffs_batch + f * num_pos;

        /* RGB to grayscale */
        rgb_to_gray(in->data, gray, pixels);

        /* DWT forward */
        dwt_forward(gray, w, h, wema->coeffs);

        /* Extract to flat array */
        extract_coeffs_flat(wema->coeffs, coeffs);
    }

    /*========================================================================
     * PHASE 2: Temporal IIR filtering (parallel across positions)
     *=======================================================================*/
    iir_temporal_process_batch(wema->iir_filt,
                               batch->coeffs_batch,
                               batch->delta_batch,
                               frame_count);

    /* Track warmup across batches */
    int warmup_done = wema->iir_filt->warmup_frames;
    int frames_start = wema->iir_filt->frames_processed - frame_count;
    int valid_start = (frames_start < warmup_done) ? (warmup_done - frames_start) : 0;
    if (valid_start > frame_count) valid_start = frame_count;

    /*========================================================================
     * PHASE 3: Amplification + IDWT for all frames (sequential)
     * TODO: Parallelize by allocating per-thread DWT structures
     *=======================================================================*/
    int output_count = 0;
    for (int f = valid_start; f < frame_count; f++) {
        const Frame *in = &frames_in[f];
        Frame *out = &frames_out[output_count];
        float *coeffs = batch->coeffs_batch + f * num_pos;
        float *delta = batch->delta_batch + f * num_pos;
        float *gray_out = batch->out_batch + f * pixels;

        /* Apply amplified delta to DWT coefficients */
        apply_delta_to_coeffs(delta, coeffs, wema->amp_factor,
                               wema->coeffs, num_pos);

        /* IDWT */
        dwt_inverse(wema->coeffs, gray_out);

        /* Grayscale to RGB */
        gray_to_rgb_simple(gray_out, in, out->data, w, h);

        output_count++;
    }

    return output_count;
}
