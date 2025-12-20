/*
 * WEMA - FFmpeg subprocess I/O
 */

#ifndef FFMPEG_IO_H
#define FFMPEG_IO_H

#include "wema.h"

/*
 * Probe video file metadata using ffprobe.
 * Returns 0 on success, -1 on error.
 */
int ffio_probe(const char *path, int *width, int *height,
               float *fps, int64_t *total_frames);

/*
 * Open FFmpeg input process.
 * Spawns ffmpeg to decode video to raw RGB24.
 * Returns 0 on success, -1 on error.
 */
int ffio_open_input(FFmpegIO *io, const char *path);

/*
 * Open FFmpeg output process.
 * Spawns ffmpeg to encode raw RGB24 to specified codec.
 * Returns 0 on success, -1 on error.
 */
int ffio_open_output(FFmpegIO *io, const char *path,
                     int width, int height, float fps,
                     const char *codec, const char *options,
                     const char *original_input);

/*
 * Read one frame from input.
 * Converts RGB24 to float [0,1].
 * Returns 1 on success, 0 on EOF, -1 on error.
 */
int ffio_read_frame(FFmpegIO *io, Frame *frame);

/*
 * Write one frame to output.
 * Converts float [0,1] to RGB24.
 * Returns 0 on success, -1 on error.
 */
int ffio_write_frame(FFmpegIO *io, const Frame *frame);

/*
 * Close input process and wait for termination.
 */
void ffio_close_input(FFmpegIO *io);

/*
 * Close output process and wait for termination.
 */
void ffio_close_output(FFmpegIO *io);

#endif /* FFMPEG_IO_H */
