/*
 * WEMA - DT-CWT Filter Coefficients
 *
 * Based on Kingsbury's near-symmetric and Q-shift filter designs.
 * References:
 *   - Kingsbury, "Complex Wavelets for Shift Invariant Analysis and
 *     Filtering of Signals", J. Applied and Computational Harmonic Analysis, 2001
 *   - Selesnick et al., "The Dual-Tree Complex Wavelet Transform",
 *     IEEE Signal Processing Magazine, 2005
 */

#ifndef DTCWT_FILTERS_H
#define DTCWT_FILTERS_H

/*
 * Near-symmetric biorthogonal filters for first level.
 * These are the antonini 7/9 filters modified for DT-CWT.
 * Length: 10 taps (symmetric-extended to 10)
 */

/* First level lowpass - Tree A */
static const float H0A_L1[] = {
    0.0f,
   -0.08838834764832f,
    0.08838834764832f,
    0.69587998903400f,
    0.69587998903400f,
    0.08838834764832f,
   -0.08838834764832f,
    0.01122679215254f,
    0.01122679215254f,
    0.0f
};

/* First level highpass - Tree A */
static const float H1A_L1[] = {
    0.01122679215254f,
    0.01122679215254f,
   -0.08838834764832f,
    0.08838834764832f,
    0.69587998903400f,
   -0.69587998903400f,
    0.08838834764832f,
    0.08838834764832f,
    0.0f,
    0.0f
};

/* First level lowpass - Tree B (shifted for approximate Hilbert pair) */
static const float H0B_L1[] = {
    0.01122679215254f,
    0.01122679215254f,
   -0.08838834764832f,
   -0.08838834764832f,
    0.69587998903400f,
    0.69587998903400f,
    0.08838834764832f,
   -0.08838834764832f,
    0.0f,
    0.0f
};

/* First level highpass - Tree B */
static const float H1B_L1[] = {
    0.0f,
    0.0f,
    0.08838834764832f,
    0.08838834764832f,
   -0.69587998903400f,
    0.69587998903400f,
    0.08838834764832f,
   -0.08838834764832f,
    0.01122679215254f,
   -0.01122679215254f
};

#define DTCWT_L1_LEN 10

/*
 * Q-shift filters for levels >= 2.
 * Quarter-sample shift design for better Hilbert pair approximation.
 * Length: 14 taps
 */

/* Q-shift lowpass - Tree A */
static const float H0A_Q[] = {
    0.00325314276365f,
   -0.00388321199280f,
   -0.03466035137290f,
    0.03887693820750f,
    0.11720388769910f,
   -0.29399291929920f,
    0.57594687138290f,
    0.57594687138290f,
   -0.29399291929920f,
    0.11720388769910f,
    0.03887693820750f,
   -0.03466035137290f,
   -0.00388321199280f,
    0.00325314276365f
};

/* Q-shift lowpass - Tree B */
static const float H0B_Q[] = {
    0.00325314276365f,
   -0.00388321199280f,
   -0.03466035137290f,
    0.03887693820750f,
    0.11720388769910f,
    0.57594687138290f,
   -0.29399291929920f,
   -0.29399291929920f,
    0.57594687138290f,
    0.11720388769910f,
    0.03887693820750f,
   -0.03466035137290f,
   -0.00388321199280f,
    0.00325314276365f
};

/* Q-shift highpass derived from lowpass via QMF */
static const float H1A_Q[] = {
    0.00325314276365f,
    0.00388321199280f,
   -0.03466035137290f,
   -0.03887693820750f,
    0.11720388769910f,
    0.29399291929920f,
    0.57594687138290f,
   -0.57594687138290f,
   -0.29399291929920f,
    0.11720388769910f,
   -0.03887693820750f,
   -0.03466035137290f,
    0.00388321199280f,
    0.00325314276365f
};

static const float H1B_Q[] = {
    0.00325314276365f,
    0.00388321199280f,
   -0.03466035137290f,
   -0.03887693820750f,
    0.11720388769910f,
   -0.57594687138290f,
    0.29399291929920f,
    0.29399291929920f,
   -0.57594687138290f,
    0.11720388769910f,
   -0.03887693820750f,
   -0.03466035137290f,
    0.00388321199280f,
    0.00325314276365f
};

#define DTCWT_Q_LEN 14

/*
 * Synthesis filters (for inverse transform).
 * For perfect reconstruction: G = reverse(H) with sign alternation on highpass.
 */

/* Synthesis lowpass - Tree A, Level 1 */
static const float G0A_L1[] = {
    0.0f,
    0.01122679215254f,
    0.01122679215254f,
   -0.08838834764832f,
    0.08838834764832f,
    0.69587998903400f,
    0.69587998903400f,
    0.08838834764832f,
   -0.08838834764832f,
    0.0f
};

/* Synthesis highpass - Tree A, Level 1 */
static const float G1A_L1[] = {
    0.0f,
    0.0f,
    0.08838834764832f,
    0.08838834764832f,
   -0.69587998903400f,
   -0.69587998903400f,
    0.08838834764832f,
   -0.08838834764832f,
    0.01122679215254f,
   -0.01122679215254f
};

/* Synthesis lowpass - Tree B, Level 1 */
static const float G0B_L1[] = {
    0.0f,
    0.0f,
   -0.08838834764832f,
    0.08838834764832f,
    0.69587998903400f,
    0.69587998903400f,
   -0.08838834764832f,
   -0.08838834764832f,
    0.01122679215254f,
    0.01122679215254f
};

/* Synthesis highpass - Tree B, Level 1 */
static const float G1B_L1[] = {
   -0.01122679215254f,
    0.01122679215254f,
   -0.08838834764832f,
    0.08838834764832f,
    0.69587998903400f,
   -0.69587998903400f,
    0.08838834764832f,
    0.08838834764832f,
    0.0f,
    0.0f
};

/* Q-shift synthesis filters (reversed analysis) */
static const float G0A_Q[] = {
    0.00325314276365f,
   -0.00388321199280f,
   -0.03466035137290f,
    0.11720388769910f,
   -0.29399291929920f,
    0.57594687138290f,
    0.57594687138290f,
   -0.29399291929920f,
    0.11720388769910f,
   -0.03466035137290f,
    0.03887693820750f,
    0.03887693820750f,
   -0.00388321199280f,
    0.00325314276365f
};

static const float G0B_Q[] = {
    0.00325314276365f,
   -0.00388321199280f,
   -0.03466035137290f,
    0.11720388769910f,
    0.57594687138290f,
   -0.29399291929920f,
   -0.29399291929920f,
    0.57594687138290f,
    0.11720388769910f,
   -0.03466035137290f,
    0.03887693820750f,
    0.03887693820750f,
   -0.00388321199280f,
    0.00325314276365f
};

static const float G1A_Q[] = {
    0.00325314276365f,
    0.00388321199280f,
   -0.03466035137290f,
    0.11720388769910f,
   -0.29399291929920f,
   -0.57594687138290f,
    0.57594687138290f,
    0.29399291929920f,
   -0.11720388769910f,
   -0.03466035137290f,
   -0.03887693820750f,
    0.03887693820750f,
    0.00388321199280f,
   -0.00325314276365f
};

static const float G1B_Q[] = {
   -0.00325314276365f,
    0.00388321199280f,
    0.03887693820750f,
   -0.03887693820750f,
   -0.03466035137290f,
   -0.11720388769910f,
    0.29399291929920f,
    0.57594687138290f,
   -0.57594687138290f,
   -0.29399291929920f,
    0.11720388769910f,
   -0.03466035137290f,
    0.00388321199280f,
    0.00325314276365f
};

#endif /* DTCWT_FILTERS_H */
