/*
 * WEMA - Phase Amplification
 */

#ifndef PHASE_AMP_H
#define PHASE_AMP_H

#include "wema.h"

/*
 * Apply phase amplification to DT-CWT coefficients.
 *
 * delta_phi: band-pass filtered phase variation [num_positions]
 * amplitude: coefficient amplitudes [num_positions]
 * alpha: amplification factor
 * coeffs: DT-CWT coefficients to modify in-place
 */
void phase_amplify(const float *delta_phi,
                   const float *amplitude,
                   float alpha,
                   DTCWTCoeffs *coeffs);

/*
 * Optional amplitude thresholding.
 * Suppresses amplification in low-amplitude (noisy) regions.
 * threshold: minimum amplitude to amplify (0.0 to disable)
 */
void phase_amplify_threshold(const float *delta_phi,
                             const float *amplitude,
                             float alpha,
                             float threshold,
                             DTCWTCoeffs *coeffs);

#endif /* PHASE_AMP_H */
