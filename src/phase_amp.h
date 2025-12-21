/*
 * WEMA - Phase Amplification
 */

#ifndef PHASE_AMP_H
#define PHASE_AMP_H

#include "wema.h"

/*
 * Apply coefficient amplification to DT-CWT coefficients.
 *
 * delta_coeff: band-pass filtered coefficient variation [num_positions]
 * orig_coeff: original coefficient values [num_positions]
 * alpha: amplification factor
 * coeffs: DT-CWT coefficients to modify in-place
 * spatial_smooth: apply 3x3 smoothing to delta coefficients
 */
void phase_amplify(const float *delta_coeff,
                   const float *orig_coeff,
                   float alpha,
                   DTCWTCoeffs *coeffs,
                   bool spatial_smooth);

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
