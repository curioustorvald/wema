/*
 * Async I/O - Double-buffered pipeline implementation
 */

#include "async_io.h"
#include "alloc.h"

#include <stdio.h>
#include <string.h>

/*
 * Reader thread: continuously reads frames into available slots
 */
static void *reader_thread_func(void *arg) {
    AsyncIOContext *ctx = (AsyncIOContext *)arg;
    int current_slot = 0;

    while (1) {
        pthread_mutex_lock(&ctx->mutex);

        /* Wait for an empty slot or shutdown */
        while (!ctx->shutdown && ctx->slots[current_slot].state != SLOT_EMPTY) {
            pthread_cond_wait(&ctx->cond_reader, &ctx->mutex);
        }

        if (ctx->shutdown) {
            pthread_mutex_unlock(&ctx->mutex);
            break;
        }

        /* Mark slot as filling */
        ctx->slots[current_slot].state = SLOT_FILLING;
        pthread_mutex_unlock(&ctx->mutex);

        /* Read frames into this slot (outside lock - I/O is slow) */
        BufferSlot *slot = &ctx->slots[current_slot];
        int frames_read = 0;
        bool eof = false;
        bool error = false;

        for (int i = 0; i < ctx->batch_size; i++) {
            int ret = ffio_read_frame(ctx->io_in, &slot->frames_in[i]);
            if (ret == 1) {
                frames_read++;
            } else if (ret == 0) {
                eof = true;
                break;
            } else {
                error = true;
                break;
            }
        }

        /* Update state */
        pthread_mutex_lock(&ctx->mutex);

        slot->frame_count = frames_read;
        ctx->total_frames_read += frames_read;

        if (error) {
            ctx->error = true;
            ctx->reader_done = true;
            slot->state = SLOT_EMPTY;
            pthread_cond_broadcast(&ctx->cond_main);
            pthread_mutex_unlock(&ctx->mutex);
            break;
        }

        if (frames_read > 0) {
            slot->state = SLOT_READY;
            pthread_cond_signal(&ctx->cond_main);
        } else {
            slot->state = SLOT_EMPTY;
        }

        if (eof) {
            ctx->reader_done = true;
            pthread_cond_broadcast(&ctx->cond_main);
            pthread_mutex_unlock(&ctx->mutex);
            break;
        }

        pthread_mutex_unlock(&ctx->mutex);

        /* Alternate between slots */
        current_slot = 1 - current_slot;
    }

    return NULL;
}

/*
 * Writer thread: writes output frames from completed slots
 */
static void *writer_thread_func(void *arg) {
    AsyncIOContext *ctx = (AsyncIOContext *)arg;

    while (1) {
        pthread_mutex_lock(&ctx->mutex);

        /* Wait for a slot with output to write, or shutdown */
        int slot_to_write = -1;
        while (!ctx->shutdown) {
            /* Check both slots for DONE state */
            for (int i = 0; i < 2; i++) {
                if (ctx->slots[i].state == SLOT_DONE) {
                    slot_to_write = i;
                    break;
                }
            }
            if (slot_to_write >= 0) break;

            /* If reader is done and no slots to write, we're done */
            if (ctx->reader_done) {
                bool any_pending = false;
                for (int i = 0; i < 2; i++) {
                    if (ctx->slots[i].state == SLOT_PROCESSING ||
                        ctx->slots[i].state == SLOT_READY ||
                        ctx->slots[i].state == SLOT_DONE) {
                        any_pending = true;
                        break;
                    }
                }
                if (!any_pending) {
                    ctx->writer_done = true;
                    pthread_cond_broadcast(&ctx->cond_main);
                    pthread_mutex_unlock(&ctx->mutex);
                    return NULL;
                }
            }

            pthread_cond_wait(&ctx->cond_writer, &ctx->mutex);
        }

        if (ctx->shutdown && slot_to_write < 0) {
            pthread_mutex_unlock(&ctx->mutex);
            break;
        }

        /* Mark slot as writing */
        ctx->slots[slot_to_write].state = SLOT_WRITING;
        pthread_mutex_unlock(&ctx->mutex);

        /* Write frames (outside lock - I/O is slow) */
        BufferSlot *slot = &ctx->slots[slot_to_write];
        bool error = false;

        for (int i = 0; i < slot->output_count; i++) {
            if (ffio_write_frame(ctx->io_out, &slot->frames_out[i]) < 0) {
                error = true;
                break;
            }
        }

        /* Update state */
        pthread_mutex_lock(&ctx->mutex);

        if (error) {
            ctx->error = true;
        } else {
            ctx->total_frames_written += slot->output_count;
        }

        /* Mark slot as empty, available for reader */
        slot->state = SLOT_EMPTY;
        slot->frame_count = 0;
        slot->output_count = 0;

        pthread_cond_signal(&ctx->cond_reader);
        pthread_cond_broadcast(&ctx->cond_main);

        if (error) {
            pthread_mutex_unlock(&ctx->mutex);
            break;
        }

        pthread_mutex_unlock(&ctx->mutex);
    }

    return NULL;
}

int async_io_init(AsyncIOContext *ctx, FFmpegIO *io_in, FFmpegIO *io_out,
                  int batch_size, bool verbose) {
    memset(ctx, 0, sizeof(*ctx));

    ctx->io_in = io_in;
    ctx->io_out = io_out;
    ctx->batch_size = batch_size;
    ctx->width = io_in->width;
    ctx->height = io_in->height;
    ctx->verbose = verbose;

    /* Initialize mutex and condition variables */
    if (pthread_mutex_init(&ctx->mutex, NULL) != 0) {
        return -1;
    }
    if (pthread_cond_init(&ctx->cond_reader, NULL) != 0) {
        pthread_mutex_destroy(&ctx->mutex);
        return -1;
    }
    if (pthread_cond_init(&ctx->cond_main, NULL) != 0) {
        pthread_cond_destroy(&ctx->cond_reader);
        pthread_mutex_destroy(&ctx->mutex);
        return -1;
    }
    if (pthread_cond_init(&ctx->cond_writer, NULL) != 0) {
        pthread_cond_destroy(&ctx->cond_main);
        pthread_cond_destroy(&ctx->cond_reader);
        pthread_mutex_destroy(&ctx->mutex);
        return -1;
    }

    /* Allocate double buffers */
    for (int s = 0; s < 2; s++) {
        BufferSlot *slot = &ctx->slots[s];
        slot->state = SLOT_EMPTY;
        slot->frame_count = 0;
        slot->output_count = 0;

        slot->frames_in = mem_alloc(batch_size * sizeof(Frame));
        slot->frames_out = mem_alloc(batch_size * sizeof(Frame));
        if (!slot->frames_in || !slot->frames_out) {
            goto alloc_error;
        }

        for (int i = 0; i < batch_size; i++) {
            memset(&slot->frames_in[i], 0, sizeof(Frame));
            memset(&slot->frames_out[i], 0, sizeof(Frame));
            if (frame_alloc(&slot->frames_in[i], ctx->width, ctx->height, 3) < 0 ||
                frame_alloc(&slot->frames_out[i], ctx->width, ctx->height, 3) < 0) {
                goto alloc_error;
            }
        }
    }

    /* Start reader and writer threads */
    if (pthread_create(&ctx->reader_thread, NULL, reader_thread_func, ctx) != 0) {
        goto alloc_error;
    }

    if (pthread_create(&ctx->writer_thread, NULL, writer_thread_func, ctx) != 0) {
        ctx->shutdown = true;
        pthread_cond_broadcast(&ctx->cond_reader);
        pthread_join(ctx->reader_thread, NULL);
        goto alloc_error;
    }

    return 0;

alloc_error:
    for (int s = 0; s < 2; s++) {
        BufferSlot *slot = &ctx->slots[s];
        if (slot->frames_in) {
            for (int i = 0; i < batch_size; i++) {
                frame_free(&slot->frames_in[i]);
            }
            mem_free(slot->frames_in);
        }
        if (slot->frames_out) {
            for (int i = 0; i < batch_size; i++) {
                frame_free(&slot->frames_out[i]);
            }
            mem_free(slot->frames_out);
        }
    }
    pthread_cond_destroy(&ctx->cond_writer);
    pthread_cond_destroy(&ctx->cond_main);
    pthread_cond_destroy(&ctx->cond_reader);
    pthread_mutex_destroy(&ctx->mutex);
    return -1;
}

int async_io_get_input_batch(AsyncIOContext *ctx, int *slot_idx) {
    pthread_mutex_lock(&ctx->mutex);

    /* Wait for a ready slot or completion */
    while (1) {
        /* Check for ready slots */
        for (int i = 0; i < 2; i++) {
            if (ctx->slots[i].state == SLOT_READY) {
                ctx->slots[i].state = SLOT_PROCESSING;
                *slot_idx = i;
                int count = ctx->slots[i].frame_count;
                pthread_mutex_unlock(&ctx->mutex);
                return count;
            }
        }

        /* Check for completion or error */
        if (ctx->error) {
            pthread_mutex_unlock(&ctx->mutex);
            return -1;
        }

        if (ctx->reader_done) {
            /* Check if there's still data being processed that might produce output */
            bool any_data = false;
            for (int i = 0; i < 2; i++) {
                if (ctx->slots[i].state != SLOT_EMPTY) {
                    any_data = true;
                    break;
                }
            }
            if (!any_data) {
                pthread_mutex_unlock(&ctx->mutex);
                return 0;  /* EOF */
            }
        }

        /* Wait for state change */
        pthread_cond_wait(&ctx->cond_main, &ctx->mutex);
    }
}

void async_io_batch_done(AsyncIOContext *ctx, int slot_idx, int output_count) {
    pthread_mutex_lock(&ctx->mutex);

    ctx->slots[slot_idx].output_count = output_count;
    ctx->slots[slot_idx].state = SLOT_DONE;

    /* Signal writer thread */
    pthread_cond_signal(&ctx->cond_writer);

    pthread_mutex_unlock(&ctx->mutex);
}

int async_io_finish(AsyncIOContext *ctx) {
    pthread_mutex_lock(&ctx->mutex);

    /* Wait for writer to finish */
    while (!ctx->writer_done && !ctx->error) {
        pthread_cond_wait(&ctx->cond_main, &ctx->mutex);
    }

    bool had_error = ctx->error;

    /* Signal shutdown */
    ctx->shutdown = true;
    pthread_cond_broadcast(&ctx->cond_reader);
    pthread_cond_broadcast(&ctx->cond_writer);

    pthread_mutex_unlock(&ctx->mutex);

    /* Join threads */
    pthread_join(ctx->reader_thread, NULL);
    pthread_join(ctx->writer_thread, NULL);

    return had_error ? -1 : 0;
}

void async_io_free(AsyncIOContext *ctx) {
    /* Free buffers */
    for (int s = 0; s < 2; s++) {
        BufferSlot *slot = &ctx->slots[s];
        if (slot->frames_in) {
            for (int i = 0; i < ctx->batch_size; i++) {
                frame_free(&slot->frames_in[i]);
            }
            mem_free(slot->frames_in);
            slot->frames_in = NULL;
        }
        if (slot->frames_out) {
            for (int i = 0; i < ctx->batch_size; i++) {
                frame_free(&slot->frames_out[i]);
            }
            mem_free(slot->frames_out);
            slot->frames_out = NULL;
        }
    }

    /* Destroy synchronization primitives */
    pthread_cond_destroy(&ctx->cond_writer);
    pthread_cond_destroy(&ctx->cond_main);
    pthread_cond_destroy(&ctx->cond_reader);
    pthread_mutex_destroy(&ctx->mutex);
}

int async_io_frames_read(const AsyncIOContext *ctx) {
    return ctx->total_frames_read;
}

int async_io_frames_written(const AsyncIOContext *ctx) {
    return ctx->total_frames_written;
}
