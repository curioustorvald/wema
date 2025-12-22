/*
 * WEMA - Wavelet-based Eulerian Motion Amplification
 * Core types and constants
 */

#ifndef WEMA_H
#define WEMA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/*============================================================================
 * Configuration Constants
 *===========================================================================*/

#define WEMA_MAX_LEVELS          4    /* Maximum 2D DWT decomposition levels */
#define WEMA_NUM_ORIENTATIONS    6    /* Subband slots per level (3 unique + 3 mirrored) */
#define WEMA_TEMPORAL_WINDOW_DEF 32   /* Default sliding window size */
#define WEMA_TEMPORAL_WINDOW_MAX 256  /* Maximum sliding window size */

/*============================================================================
 * Video Frame
 *===========================================================================*/

typedef struct {
    float *data;      /* Pixel data [height * width * channels] */
    int    width;
    int    height;
    int    channels;  /* 1=grayscale, 3=RGB */
} Frame;

/*============================================================================
 * DWT Subband (single orientation at single level)
 *===========================================================================*/

typedef struct {
    float   *coeffs;  /* Wavelet coefficients [height * width] */
    int      width;   /* Subband width (halved at each level) */
    int      height;  /* Subband height (halved at each level) */
} Subband;

/*============================================================================
 * 2D DWT Decomposition (single frame)
 *===========================================================================*/

typedef struct {
    /* Lowpass residual */
    float *lowpass;
    int    lowpass_w;
    int    lowpass_h;

    /* Highpass subbands: [level][orientation] */
    Subband subbands[WEMA_MAX_LEVELS][WEMA_NUM_ORIENTATIONS];

    int num_levels;
    int orig_width;
    int orig_height;
} DWTCoeffs;

/*============================================================================
 * Temporal Sliding Window Buffer
 *===========================================================================*/

typedef struct {
    float  *phase_buffer;   /* Circular buffer [num_positions * window_size] */
    float  *amplitude;      /* Most recent amplitude [num_positions] */
    float  *dc_phase;       /* DC phase (mean) [num_positions] */

    size_t  num_positions;  /* Total coefficient positions */
    int     window_size;    /* Temporal window size */
    int     head;           /* Current head in circular buffer */
    int     filled;         /* Number of valid frames */

    /* Subband dimensions for indexing */
    int     num_levels;
    int     widths[WEMA_MAX_LEVELS];
    int     heights[WEMA_MAX_LEVELS];
} TemporalBuffer;

/*============================================================================
 * Temporal Wavelet Filter State
 *===========================================================================*/

typedef struct {
    float  *lp_filter;      /* Lowpass filter coefficients */
    float  *hp_filter;      /* Highpass filter coefficients */
    int     filter_len;

    float   f_low_norm;     /* Low cutoff (normalized to sample rate) */
    float   f_high_norm;    /* High cutoff (normalized to sample rate) */

    int     temporal_levels;
} TemporalFilter;

/*============================================================================
 * WEMA Processing Context
 *===========================================================================*/

typedef struct {
    /* Video parameters */
    int    width;
    int    height;
    float  fps;

    /* Processing parameters */
    float  amp_factor;      /* Amplification factor (alpha) */
    float  f_low;           /* Low frequency cutoff (Hz) */
    float  f_high;          /* High frequency cutoff (Hz) */
    int    num_levels;      /* DWT decomposition levels */
    int    temporal_window; /* Temporal sliding window size */
    bool   edge_aware;      /* Use edge-aware guided filter */
    bool   bilateral_temp;  /* Bilateral temporal filtering */
    bool   color_mode;      /* Process chrominance (Cb/Cr) channels */

    /* Normalized frequencies */
    float  f_low_norm;
    float  f_high_norm;

    /* Internal state - luminance (Y) */
    DWTCoeffs      *coeffs;
    TemporalBuffer *temporal_buf;
    TemporalFilter *temporal_filt;

    /* Internal state - chrominance (Cb, Cr) for color mode */
    DWTCoeffs      *coeffs_cb;
    DWTCoeffs      *coeffs_cr;
    TemporalBuffer *temporal_buf_cb;
    TemporalBuffer *temporal_buf_cr;

    /* Work buffers - luminance */
    float  *gray_in;        /* Y channel input */
    float  *gray_out;       /* Y channel output */
    float  *delta_phi;      /* Filtered coefficient delta (Y) */

    /* Work buffers - chrominance */
    float  *cb_in;          /* Cb channel input */
    float  *cr_in;          /* Cr channel input */
    float  *cb_out;         /* Cb channel output */
    float  *cr_out;         /* Cr channel output */
    float  *delta_cb;       /* Filtered coefficient delta (Cb) */
    float  *delta_cr;       /* Filtered coefficient delta (Cr) */

    /* Shared work buffers */
    float  *delta_buf;      /* Delta for spatial smoothing */
    float  *smooth_buf;     /* Smoothed delta output */
    float  *guide_buf;      /* Guide image for edge-aware filter */
    float  *guided_work[4]; /* Work buffers for guided filter */
} WemaContext;

/*============================================================================
 * FFmpeg I/O Context
 *===========================================================================*/

typedef struct {
    /* Input process */
    int     in_pid;
    int     in_fd;

    /* Output process */
    int     out_pid;
    int     out_fd;

    /* Video info */
    int     width;
    int     height;
    float   fps;
    int64_t total_frames;

    char   *input_path;

    /* Frame buffer */
    uint8_t *raw_buffer;
    size_t   buffer_size;
} FFmpegIO;

/*============================================================================
 * CLI Configuration
 *===========================================================================*/

typedef struct {
    char   *input_path;
    char   *output_path;

    float   amp_factor;     /* Default: 50.0 */
    float   f_low;          /* Default: 0.5 Hz */
    float   f_high;         /* Default: 3.0 Hz */
    int     temporal_window;/* Default: 32, higher = better freq resolution */

    char   *ff_codec;       /* Default: "ffv1" */
    char   *ff_options;     /* Default: NULL */

    bool    verbose;
    bool    edge_aware;     /* Default: true (use guided filter) */
    bool    bilateral_temp; /* Default: true (bilateral temporal filtering) */
    bool    color_mode;     /* Default: false (amplify chrominance for blood flow) */
} WemaConfig;

/*============================================================================
 * Function Prototypes - wema.c
 *===========================================================================*/

int  wema_init(WemaContext *ctx, int width, int height, float fps,
               float amp_factor, float f_low, float f_high,
               int temporal_window);
int  wema_init_color(WemaContext *ctx);  /* Init chrominance for color mode */
void wema_free(WemaContext *ctx);
int  wema_process_frame(WemaContext *ctx, const Frame *in, Frame *out);
int  wema_flush(WemaContext *ctx, Frame *out);
bool wema_ready(const WemaContext *ctx);

/*============================================================================
 * Function Prototypes - Frame utilities
 *===========================================================================*/

int  frame_alloc(Frame *frame, int width, int height, int channels);
void frame_free(Frame *frame);
void frame_rgb_to_gray(const Frame *rgb, float *gray);
void frame_gray_to_rgb(const float *gray, const Frame *orig, Frame *out,
                       float *delta_buf, float *smooth_buf);

#endif /* WEMA_H */
