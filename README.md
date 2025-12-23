# WEMA - Wavelet-based Eulerian Motion Amplification

A pure C implementation of Eulerian video motion amplification using 2D discrete wavelet transforms.

WEMA reveals subtle motions in video that are invisible to the naked eye by amplifying small temporal variations in wavelet coefficients within a user-specified frequency band.

## Features

- **CDF 9/7 biorthogonal wavelets** - JPEG 2000 wavelet for smooth reconstruction
- **Temporal bandpass filtering** - Frequency isolation using 2nd-order butterworth filter
- **Bilateral temporal filtering** - Noise reduction while preserving motion coherence
- **Edge-aware guided filter** - Reduces artifacts at edges during reconstruction
- **Optimized for throughput** - Vectorization-friendly code with `-Ofast -march=native`
- **Pure C11** - No external dependencies except FFmpeg for I/O

## Requirements

- GCC or Clang with C11 support
- FFmpeg (for video I/O)
- POSIX-compatible system (Linux, macOS, WSL)

## Building

```bash
# Release build (optimized)
make all

# Debug build (with AddressSanitizer)
make debug

# Install to /usr/local/bin
sudo make install
```

## Usage

```bash
# Basic usage - amplify motion in 0.5-3.0 Hz band by 50x
wema -i input.mp4

# Custom frequency band and amplification
wema -i input.mp4 -o output.mkv -a 100 --fl 0.8 --fh 2.0

# Disable denoising features for raw output
wema -i input.mp4 --no-edge-aware

# Output to H.264 with quality setting
wema -i input.mp4 -o output.mp4 --ff-codec libx264 --ff-option '-crf 18'

# Verbose output
wema -i input.mp4 -v
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input <file>` | Input video file (required) | - |
| `-o, --output <file>` | Output video file | `input.wema.mkv` |
| `-a, --amp <factor>` | Amplification factor | 120 |
| `--fl <freq>` | Low frequency cutoff (Hz) | 0.5 |
| `--fh <freq>` | High frequency cutoff (Hz) | half of input video framerate |
| `--temporal-window <n>` | Temporal window size (4-256) | 32 |
| `--no-edge-aware` | Disable edge-aware guided filter | enabled |
| `--bilateral-filter` | Enable bilateral temporal filtering | disabled |
| `--color` | Enable color difference amplification | disabled |
| `--ff-codec <codec>` | FFmpeg video codec | ffv1 |
| `--ff-option <opts>` | FFmpeg encoder options | - |
| `-v, --verbose` | Verbose output | off |
| `-h, --help` | Show help | - |

## How It Works

1. **Spatial Decomposition** - Each frame is decomposed using a 2D CDF 9/7 discrete wavelet transform into lowpass and highpass subbands (LH, HL, HH) at multiple scales.

2. **Temporal Filtering** - Wavelet coefficients are tracked over time in a sliding window. A butterworth temporal bandpass filter isolates motion in the desired frequency range.

3. **Bilateral Denoising** - Optionally, temporal samples are weighted by similarity to the current frame, preserving coherent motion while averaging out sensor noise.

4. **Coefficient Amplification** - Bandpass-filtered coefficient deltas are amplified and added back to the original coefficients. Adaptive thresholding prevents amplification of noise in low-energy regions.

5. **Reconstruction** - Inverse DWT reconstructs the amplified grayscale image. An edge-aware guided filter smooths the amplification signal while preserving edges.

6. **Color Transfer** - The amplified luminance delta is added to the original RGB frame.

## Example Applications

- **Structural health monitoring** - Visualize building/bridge vibrations
- **Medical imaging** - Detect pulse, breathing, blood flow
- **Industrial inspection** - Reveal machine vibrations
- **Scientific visualization** - Observe subtle physical phenomena

## References

- Wu, H.-Y., et al. "Eulerian Video Magnification for Revealing Subtle Changes in the World." ACM SIGGRAPH 2012.

## License

MIT License
