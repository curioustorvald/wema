/*
 * WEMA - FFmpeg subprocess I/O
 */

#include "ffmpeg_io.h"
#include "alloc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <errno.h>

/* Read entire output from a command */
static char *read_command_output(const char *cmd) {
    FILE *fp = popen(cmd, "r");
    if (!fp) return NULL;

    size_t capacity = 4096;
    size_t len = 0;
    char *buf = mem_alloc(capacity);
    if (!buf) {
        pclose(fp);
        return NULL;
    }

    while (1) {
        size_t n = fread(buf + len, 1, capacity - len - 1, fp);
        if (n == 0) break;
        len += n;
        if (len + 1 >= capacity) {
            capacity *= 2;
            char *newbuf = realloc(buf, capacity);
            if (!newbuf) {
                mem_free(buf);
                pclose(fp);
                return NULL;
            }
            buf = newbuf;
        }
    }
    buf[len] = '\0';
    pclose(fp);
    return buf;
}

int ffio_probe(const char *path, int *width, int *height,
               float *fps, int64_t *total_frames) {
    char cmd[2048];

    /* Get video stream info as JSON */
    snprintf(cmd, sizeof(cmd),
        "ffprobe -v quiet -select_streams v:0 "
        "-show_entries stream=width,height,r_frame_rate,nb_frames "
        "-of csv=p=0 \"%s\" 2>/dev/null",
        path);

    char *output = read_command_output(cmd);
    if (!output) return -1;

    /* Parse CSV: width,height,fps_num/fps_den,nb_frames */
    int w = 0, h = 0;
    int fps_num = 0, fps_den = 1;
    int64_t frames = -1;

    char *line = output;
    char *tok = strtok(line, ",");
    if (tok) w = atoi(tok);
    tok = strtok(NULL, ",");
    if (tok) h = atoi(tok);
    tok = strtok(NULL, ",");
    if (tok) {
        char *slash = strchr(tok, '/');
        if (slash) {
            fps_num = atoi(tok);
            fps_den = atoi(slash + 1);
        } else {
            fps_num = atoi(tok);
            fps_den = 1;
        }
    }
    tok = strtok(NULL, ",\n");
    if (tok && strcmp(tok, "N/A") != 0) {
        frames = atoll(tok);
    }

    mem_free(output);

    if (w <= 0 || h <= 0 || fps_num <= 0 || fps_den <= 0) {
        return -1;
    }

    *width = w;
    *height = h;
    *fps = (float)fps_num / (float)fps_den;
    *total_frames = frames;

    return 0;
}

int ffio_open_input(FFmpegIO *io, const char *path) {
    memset(io, 0, sizeof(*io));
    io->in_pid = -1;
    io->out_pid = -1;
    io->in_fd = -1;
    io->out_fd = -1;

    /* Probe first */
    if (ffio_probe(path, &io->width, &io->height, &io->fps, &io->total_frames) < 0) {
        return -1;
    }

    io->input_path = strdup(path);
    if (!io->input_path) return -1;

    /* Allocate frame buffer */
    io->buffer_size = (size_t)io->width * io->height * 3;
    io->raw_buffer = mem_alloc(io->buffer_size);
    if (!io->raw_buffer) {
        free(io->input_path);
        return -1;
    }

    /* Create pipe */
    int pipefd[2];
    if (pipe(pipefd) < 0) {
        mem_free(io->raw_buffer);
        free(io->input_path);
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        mem_free(io->raw_buffer);
        free(io->input_path);
        return -1;
    }

    if (pid == 0) {
        /* Child: ffmpeg decoder */
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);

        /* Redirect stderr to /dev/null */
        freopen("/dev/null", "w", stderr);

        execlp("ffmpeg", "ffmpeg",
               "-hide_banner",
               "-i", path,
               "-f", "rawvideo",
               "-pix_fmt", "rgb24",
               "-",
               (char *)NULL);
        _exit(1);
    }

    /* Parent */
    close(pipefd[1]);
    io->in_fd = pipefd[0];
    io->in_pid = pid;

    return 0;
}

int ffio_open_output(FFmpegIO *io, const char *path,
                     int width, int height, float fps,
                     const char *codec, const char *options,
                     const char *original_input) {
    io->out_pid = -1;
    io->out_fd = -1;

    /* Create pipe */
    int pipefd[2];
    if (pipe(pipefd) < 0) {
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        return -1;
    }

    if (pid == 0) {
        /* Child: ffmpeg encoder */
        close(pipefd[1]);
        dup2(pipefd[0], STDIN_FILENO);
        close(pipefd[0]);

        /* Redirect stderr to /dev/null for clean output */
        freopen("/dev/null", "w", stderr);

        /* Build argument list */
        char size_str[32];
        char fps_str[32];
        snprintf(size_str, sizeof(size_str), "%dx%d", width, height);
        snprintf(fps_str, sizeof(fps_str), "%.6f", fps);

        const char *actual_codec = codec ? codec : "ffv1";

        if (original_input && options) {
            execlp("ffmpeg", "ffmpeg",
                   "-hide_banner",
                   "-f", "rawvideo",
                   "-pix_fmt", "rgb24",
                   "-s", size_str,
                   "-r", fps_str,
                   "-i", "-",
                   "-i", original_input,
                   "-map", "0:v",
                   "-map", "1:a?",
                   "-c:a", "copy",
                   "-c:v", actual_codec,
                   options,
                   "-y", path,
                   (char *)NULL);
        } else if (original_input) {
            execlp("ffmpeg", "ffmpeg",
                   "-hide_banner",
                   "-f", "rawvideo",
                   "-pix_fmt", "rgb24",
                   "-s", size_str,
                   "-r", fps_str,
                   "-i", "-",
                   "-i", original_input,
                   "-map", "0:v",
                   "-map", "1:a?",
                   "-c:a", "copy",
                   "-c:v", actual_codec,
                   "-y", path,
                   (char *)NULL);
        } else {
            execlp("ffmpeg", "ffmpeg",
                   "-hide_banner",
                   "-f", "rawvideo",
                   "-pix_fmt", "rgb24",
                   "-s", size_str,
                   "-r", fps_str,
                   "-i", "-",
                   "-c:v", actual_codec,
                   "-y", path,
                   (char *)NULL);
        }
        _exit(1);
    }

    /* Parent */
    close(pipefd[0]);
    io->out_fd = pipefd[1];
    io->out_pid = pid;

    /* Ignore SIGPIPE */
    signal(SIGPIPE, SIG_IGN);

    return 0;
}

int ffio_read_frame(FFmpegIO *io, Frame *frame) {
    if (io->in_fd < 0) return -1;

    size_t to_read = io->buffer_size;
    size_t total = 0;
    uint8_t *ptr = io->raw_buffer;

    while (total < to_read) {
        ssize_t n = read(io->in_fd, ptr + total, to_read - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            /* EOF */
            if (total == 0) return 0;
            return -1; /* Partial frame */
        }
        total += (size_t)n;
    }

    /* Convert RGB24 uint8 to float [0,1] */
    if (!frame->data) {
        if (frame_alloc(frame, io->width, io->height, 3) < 0) {
            return -1;
        }
    }

    for (size_t i = 0; i < io->buffer_size; i++) {
        frame->data[i] = (float)io->raw_buffer[i] / 255.0f;
    }

    return 1;
}

int ffio_write_frame(FFmpegIO *io, const Frame *frame) {
    if (io->out_fd < 0) return -1;

    size_t size = (size_t)frame->width * frame->height * frame->channels;

    /* Convert float [0,1] to RGB24 uint8 */
    for (size_t i = 0; i < size; i++) {
        float val = frame->data[i];
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
        io->raw_buffer[i] = (uint8_t)(val * 255.0f + 0.5f);
    }

    size_t total = 0;
    while (total < size) {
        ssize_t n = write(io->out_fd, io->raw_buffer + total, size - total);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        total += (size_t)n;
    }

    return 0;
}

void ffio_close_input(FFmpegIO *io) {
    if (io->in_fd >= 0) {
        close(io->in_fd);
        io->in_fd = -1;
    }
    if (io->in_pid > 0) {
        waitpid(io->in_pid, NULL, 0);
        io->in_pid = -1;
    }
    if (io->raw_buffer) {
        mem_free(io->raw_buffer);
        io->raw_buffer = NULL;
    }
    if (io->input_path) {
        free(io->input_path);
        io->input_path = NULL;
    }
}

void ffio_close_output(FFmpegIO *io) {
    if (io->out_fd >= 0) {
        close(io->out_fd);
        io->out_fd = -1;
    }
    if (io->out_pid > 0) {
        waitpid(io->out_pid, NULL, 0);
        io->out_pid = -1;
    }
}
