## WEMA: Wavelet-based Eulerian Motion Amplification

* Implementation must be written in C (no C++)
* Address Sanitizer for `make debug`
* Video I/O: spawn FFmpeg process
* Video output: FFV1/arbitrary
* Audio output: copy from input (`-c:a copy`)

### Commandline

```
./wema -i input.mp4 -o output.mkv  // outputs FFV1

./wema -i input.mp4 -o output.mp4 --ff-codec libx264 --ff-option '-crf 32'  // outputs mp4 via FFmpeg with -c:v libx264 -crf 32

./wema -i input.mp4  // autogenerate output file (input.wema.mkv), outputs FFV1
```

#### FFmpeg controller arguments

* `--ff-codec`: fed into `-c:v`
* `--ff-option`: encoder options
