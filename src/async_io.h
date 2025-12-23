/*
 * Async I/O - Double-buffered pipeline for overlapping I/O with computation
 *
 * Architecture:
 *   Reader thread  →  [Buffer A/B]  →  Main thread (process)  →  [Output A/B]  →  Writer thread
 *
 * The reader thread fills the "next" input buffer while main processes "current".
 * The writer thread writes the "previous" output buffer while main processes "current".
 */

#ifndef ASYNC_IO_H
#define ASYNC_IO_H

#include "wema.h"
#include "ffmpeg_io.h"

#include <pthread.h>
#include <stdbool.h>

/*
 * Double-buffer slot states
 */
typedef enum {
    SLOT_EMPTY,      /* Buffer is empty, available for filling */
    SLOT_FILLING,    /* Reader thread is filling this buffer */
    SLOT_READY,      /* Buffer is full, ready for processing */
    SLOT_PROCESSING, /* Main thread is processing this buffer */
    SLOT_DONE,       /* Processing complete, output ready for writing */
    SLOT_WRITING     /* Writer thread is writing output */
} SlotState;

/*
 * A single buffer slot (input frames + output frames)
 */
typedef struct {
    Frame *frames_in;      /* Input frame array [batch_size] */
    Frame *frames_out;     /* Output frame array [batch_size] */
    int    frame_count;    /* Actual frames in this batch (may be < batch_size) */
    int    output_count;   /* Number of output frames after processing */
    SlotState state;       /* Current state of this slot */
} BufferSlot;

/*
 * Async I/O context
 */
typedef struct {
    /* Double buffers (ping-pong) */
    BufferSlot slots[2];
    int batch_size;
    int width, height;

    /* FFmpeg I/O handles */
    FFmpegIO *io_in;
    FFmpegIO *io_out;

    /* Threading */
    pthread_t reader_thread;
    pthread_t writer_thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond_reader;   /* Signal reader when slot becomes EMPTY */
    pthread_cond_t cond_main;     /* Signal main when slot becomes READY or writing done */
    pthread_cond_t cond_writer;   /* Signal writer when slot becomes DONE */

    /* State */
    bool reader_done;      /* Reader has finished (EOF or error) */
    bool writer_done;      /* Writer has finished all work */
    bool shutdown;         /* Signal threads to exit */
    bool error;            /* Error occurred */

    /* Statistics */
    int total_frames_read;
    int total_frames_written;

    /* Verbose flag */
    bool verbose;
} AsyncIOContext;

/*
 * Initialize async I/O context.
 * Allocates double buffers and starts reader/writer threads.
 *
 * @param ctx         Context to initialize
 * @param io_in       FFmpeg input handle (must be open)
 * @param io_out      FFmpeg output handle (must be open)
 * @param batch_size  Frames per batch
 * @param verbose     Print progress
 * @return 0 on success, -1 on error
 */
int async_io_init(AsyncIOContext *ctx, FFmpegIO *io_in, FFmpegIO *io_out,
                  int batch_size, bool verbose);

/*
 * Get the next batch of input frames for processing.
 * Blocks until frames are available or EOF.
 *
 * @param ctx         Context
 * @param slot_idx    Output: which slot (0 or 1) contains the frames
 * @return Number of frames in batch, 0 on EOF, -1 on error
 */
int async_io_get_input_batch(AsyncIOContext *ctx, int *slot_idx);

/*
 * Mark a batch as processed and ready for writing.
 * The output frames should already be in slots[slot_idx].frames_out.
 *
 * @param ctx         Context
 * @param slot_idx    Slot that was processed
 * @param output_count Number of output frames (may differ from input due to warmup)
 */
void async_io_batch_done(AsyncIOContext *ctx, int slot_idx, int output_count);

/*
 * Wait for all I/O to complete and shut down threads.
 *
 * @param ctx         Context
 * @return 0 on success, -1 on error
 */
int async_io_finish(AsyncIOContext *ctx);

/*
 * Free async I/O resources.
 * Must call async_io_finish() first.
 */
void async_io_free(AsyncIOContext *ctx);

/*
 * Get total frames read so far.
 */
int async_io_frames_read(const AsyncIOContext *ctx);

/*
 * Get total frames written so far.
 */
int async_io_frames_written(const AsyncIOContext *ctx);

#endif /* ASYNC_IO_H */
