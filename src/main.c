/*
 * WEMA - Wavelet-based Eulerian Motion Amplification
 * Main entry point and CLI parsing
 */

#include "wema.h"
#include "ffmpeg_io.h"
#include "alloc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <libgen.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "WEMA - Wavelet-based Eulerian Motion Amplification\n"
        "\n"
        "Usage: %s -i <input> [-o <output>] [options]\n"
        "\n"
        "Options:\n"
        "  -i, --input <file>     Input video file (required)\n"
        "  -o, --output <file>    Output video file (default: input.wema.mkv)\n"
        "  -a, --amp <factor>     Amplification factor (default: 10)\n"
        "  --fl <freq>            Low frequency cutoff in Hz (default: 0.5)\n"
        "  --fh <freq>            High frequency cutoff in Hz (default: 3.0)\n"
        "  --ff-codec <codec>     FFmpeg video codec (default: ffv1)\n"
        "  --ff-option <opts>     FFmpeg encoder options\n"
        "  -v, --verbose          Verbose output\n"
        "  -h, --help             Show this help\n"
        "\n"
        "Examples:\n"
        "  %s -i video.mp4\n"
        "  %s -i video.mp4 -o amplified.mkv -a 20 --fl 0.8 --fh 2.0\n"
        "  %s -i video.mp4 -o out.mp4 --ff-codec libx264 --ff-option '-crf 18'\n"
        "\n",
        prog, prog, prog, prog);
}

static char *generate_output_path(const char *input) {
    /* Generate output path: input.wema.mkv */
    size_t len = strlen(input);
    char *output = mem_alloc(len + 16);
    if (!output) return NULL;

    /* Find last dot for extension */
    const char *dot = strrchr(input, '.');
    size_t base_len = dot ? (size_t)(dot - input) : len;

    memcpy(output, input, base_len);
    strcpy(output + base_len, ".wema.mkv");

    return output;
}

static int parse_args(int argc, char **argv, WemaConfig *config) {
    /* Set defaults */
    memset(config, 0, sizeof(*config));
    config->amp_factor = 10.0f;
    config->f_low = 0.5f;
    config->f_high = 3.0f;
    config->verbose = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -i requires an argument\n");
                return -1;
            }
            config->input_path = argv[i];
        }
        else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -o requires an argument\n");
                return -1;
            }
            config->output_path = argv[i];
        }
        else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--amp") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: -a requires an argument\n");
                return -1;
            }
            config->amp_factor = strtof(argv[i], NULL);
            if (config->amp_factor <= 0) {
                fprintf(stderr, "Error: amplification factor must be positive\n");
                return -1;
            }
        }
        else if (strcmp(argv[i], "--fl") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: --fl requires an argument\n");
                return -1;
            }
            config->f_low = strtof(argv[i], NULL);
            if (config->f_low < 0) {
                fprintf(stderr, "Error: low frequency must be non-negative\n");
                return -1;
            }
        }
        else if (strcmp(argv[i], "--fh") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: --fh requires an argument\n");
                return -1;
            }
            config->f_high = strtof(argv[i], NULL);
            if (config->f_high <= 0) {
                fprintf(stderr, "Error: high frequency must be positive\n");
                return -1;
            }
        }
        else if (strcmp(argv[i], "--ff-codec") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: --ff-codec requires an argument\n");
                return -1;
            }
            config->ff_codec = argv[i];
        }
        else if (strcmp(argv[i], "--ff-option") == 0) {
            if (++i >= argc) {
                fprintf(stderr, "Error: --ff-option requires an argument\n");
                return -1;
            }
            config->ff_options = argv[i];
        }
        else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            config->verbose = true;
        }
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        else {
            fprintf(stderr, "Error: Unknown option: %s\n", argv[i]);
            return -1;
        }
    }

    if (config->input_path == NULL) {
        fprintf(stderr, "Error: Input file required (-i)\n");
        return -1;
    }

    if (config->f_low >= config->f_high) {
        fprintf(stderr, "Error: Low frequency must be less than high frequency\n");
        return -1;
    }

    return 0;
}

int main(int argc, char **argv) {
    WemaConfig config;
    char *auto_output = NULL;

    if (parse_args(argc, argv, &config) < 0) {
        print_usage(argv[0]);
        return 1;
    }

    /* Auto-generate output path if needed */
    if (config.output_path == NULL) {
        auto_output = generate_output_path(config.input_path);
        if (!auto_output) {
            fprintf(stderr, "Error: Failed to allocate output path\n");
            return 1;
        }
        config.output_path = auto_output;
    }

    /* Open input */
    FFmpegIO io_in = {0};
    if (ffio_open_input(&io_in, config.input_path) < 0) {
        fprintf(stderr, "Error: Cannot open input: %s\n", config.input_path);
        mem_free(auto_output);
        return 1;
    }

    if (config.verbose) {
        fprintf(stderr, "Input: %s\n", config.input_path);
        fprintf(stderr, "  Size: %dx%d\n", io_in.width, io_in.height);
        fprintf(stderr, "  FPS: %.3f\n", io_in.fps);
        if (io_in.total_frames > 0) {
            fprintf(stderr, "  Frames: %ld\n", (long)io_in.total_frames);
        }
        fprintf(stderr, "Output: %s\n", config.output_path);
        fprintf(stderr, "Settings:\n");
        fprintf(stderr, "  Amplification: %.1f\n", config.amp_factor);
        fprintf(stderr, "  Frequency band: %.2f - %.2f Hz\n", config.f_low, config.f_high);
    }

    /* Initialize WEMA context */
    WemaContext ctx;
    if (wema_init(&ctx, io_in.width, io_in.height, io_in.fps,
                  config.amp_factor, config.f_low, config.f_high) < 0) {
        fprintf(stderr, "Error: Failed to initialize WEMA\n");
        ffio_close_input(&io_in);
        mem_free(auto_output);
        return 1;
    }

    /* Open output */
    FFmpegIO io_out = {0};
    io_out.raw_buffer = mem_alloc((size_t)io_in.width * io_in.height * 3);
    if (!io_out.raw_buffer) {
        fprintf(stderr, "Error: Failed to allocate output buffer\n");
        wema_free(&ctx);
        ffio_close_input(&io_in);
        mem_free(auto_output);
        return 1;
    }

    if (ffio_open_output(&io_out, config.output_path,
                         io_in.width, io_in.height, io_in.fps,
                         config.ff_codec, config.ff_options,
                         config.input_path) < 0) {
        fprintf(stderr, "Error: Cannot open output: %s\n", config.output_path);
        mem_free(io_out.raw_buffer);
        wema_free(&ctx);
        ffio_close_input(&io_in);
        mem_free(auto_output);
        return 1;
    }

    /* Allocate frames */
    Frame frame_in = {0};
    Frame frame_out = {0};

    if (frame_alloc(&frame_in, io_in.width, io_in.height, 3) < 0 ||
        frame_alloc(&frame_out, io_in.width, io_in.height, 3) < 0) {
        fprintf(stderr, "Error: Failed to allocate frames\n");
        frame_free(&frame_in);
        frame_free(&frame_out);
        mem_free(io_out.raw_buffer);
        ffio_close_output(&io_out);
        wema_free(&ctx);
        ffio_close_input(&io_in);
        mem_free(auto_output);
        return 1;
    }

    /* Main processing loop */
    int frame_num = 0;
    int output_count = 0;
    int ret;

    while ((ret = ffio_read_frame(&io_in, &frame_in)) == 1) {
        frame_num++;

        /* Process frame */
        if (wema_process_frame(&ctx, &frame_in, &frame_out) == 0) {
            if (wema_ready(&ctx)) {
                if (ffio_write_frame(&io_out, &frame_out) < 0) {
                    fprintf(stderr, "Error: Failed to write frame %d\n", frame_num);
                    break;
                }
                output_count++;
            }
        }

        if (config.verbose && frame_num % 100 == 0) {
            fprintf(stderr, "\rProcessed %d frames...", frame_num);
            fflush(stderr);
        }
    }

    if (ret < 0) {
        fprintf(stderr, "Error: Failed reading frame %d\n", frame_num + 1);
    }

    /* Flush remaining frames */
    while (wema_flush(&ctx, &frame_out) == 0) {
        if (ffio_write_frame(&io_out, &frame_out) < 0) {
            fprintf(stderr, "Error: Failed to write flushed frame\n");
            break;
        }
        output_count++;
    }

    if (config.verbose) {
        fprintf(stderr, "\rProcessed %d frames, output %d frames.\n",
                frame_num, output_count);
    }

    /* Cleanup */
    frame_free(&frame_in);
    frame_free(&frame_out);
    mem_free(io_out.raw_buffer);
    ffio_close_output(&io_out);
    wema_free(&ctx);
    ffio_close_input(&io_in);
    mem_free(auto_output);

    return 0;
}
