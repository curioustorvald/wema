Below is a **minimal, wavelet-only Eulerian motion amplification sketch**, written as implementation-oriented Markdown.
This is intentionally *barebones* and maps 1-to-1 onto real code.

---

# Minimal Wavelet-Based Motion Amplification

## Assumptions

* Motion is **small** (sub-pixel, no phase wrapping)
* We amplify **temporal motion**, not appearance
* Use a **complex, approximately shift-invariant wavelet**

  * DT-CWT or steerable pyramid
* Temporal filtering is linear (band-pass)

---

## High-Level Pipeline

```text
Video
  ↓
Per-frame complex wavelet transform
  ↓
Temporal filtering of wavelet coefficients
  ↓
Phase amplification
  ↓
Inverse wavelet transform
  ↓
Amplified video
```

---

## Notation

* Frame index: `t`
* Wavelet scale: `s`
* Orientation: `o`
* Spatial position: `(x, y)`
* Complex coefficient:

  ```
  W[s,o,x,y,t] = A[s,o,x,y,t] · exp(i · φ[s,o,x,y,t])
  ```

---

## Step 1 — Spatial Wavelet Decomposition

For each frame:

```pseudo
for t in frames:
    W[:,:,:,:,t] = ComplexWaveletTransform(frame[t])
```

Requirements:

* Complex coefficients (real + imaginary)
* Near shift-invariance
* Multiple orientations

> Real wavelets technically work, but phase handling becomes painful.

---

## Step 2 — Temporal Processing (per coefficient)

For every `(s, o, x, y)`:

### 2.1 Extract phase over time

```pseudo
φ[t] = angle(W[s,o,x,y,t])
A[t] = abs(W[s,o,x,y,t])
```

### 2.2 Remove DC phase (optional but stabilizing)

```pseudo
φ0 = mean(φ[t])
Δφ[t] = φ[t] - φ0
```

---

## Step 3 — Temporal Band-Pass Filtering

Choose frequency band `[f_low, f_high]` relevant to motion:

```pseudo
Δφ_filt[t] = BandPassFilter(Δφ[t])
```

This can be:

* FFT + mask
* Temporal IIR
* Temporal wavelet (still “nothing but wavelets”)

---

## Step 4 — Phase Amplification

```pseudo
α = amplification_factor

φ_amp[t] = φ0 + α · Δφ_filt[t]
```

Constraints:

* `α` too large → phase wrapping artifacts
* Safe region:

  ```
  |α · Δφ| < π
  ```

---

## Step 5 — Reconstruct Modified Coefficients

```pseudo
W'[s,o,x,y,t] = A[t] · exp(i · φ_amp[t])
```

Magnitude is unchanged
Motion lives entirely in phase

---

## Step 6 — Inverse Wavelet Transform

For each frame:

```pseudo
frame_out[t] = InverseComplexWaveletTransform(W'[:,:,:,:,t])
```

That’s it.
No pixel warping. No flow. No meshes.

---

## Optional Stabilizers (Highly Recommended)

### Phase Unwrapping

```pseudo
φ[t] = unwrap(φ[t])
```

### Amplitude Thresholding (noise suppression)

```pseudo
if A[t] < ε:
    skip amplification
```

### Scale Selection

Ignore:

* Very fine scales → noise
* Very coarse scales → illumination drift

---

## Why This Works (One-Line Intuition)

Small motion ≈ **temporal phase modulation** of localized spatial frequencies.
Wavelets expose that modulation directly.

---

## Computational Notes

* Memory heavy: coefficients × frames
* Can stream in time with sliding window
* Easily GPU-friendly
* Cleanly integrates with codecs (your wavelet instincts are spot-on)

---

## Failure Modes (Know Them)

| Issue   | Cause                         |
| ------- | ----------------------------- |
| Haloing | Phase wrapping                |
| Shimmer | Noise in high-freq scales     |
| Drift   | DC leakage in temporal filter |
| Jelly   | Motion not small              |

---

## TL;DR

> **Motion amplification = phase EQ on a complex wavelet time series**
