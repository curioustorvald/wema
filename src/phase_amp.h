/*
 * WEMA - Coefficient Amplification
 */

#ifndef PHASE_AMP_H
#define PHASE_AMP_H

#include "wema.h"

/*
 * Apply coefficient amplification to DWT coefficients.
 *
 * delta_coeff: band-pass filtered coefficient variation [num_positions]
 * orig_coeff: original coefficient values [num_positions]
 * alpha: amplification factor
 * coeffs: DWT coefficients to modify in-place
 */
void coeff_amplify(const float *delta_coeff,
                   const float *orig_coeff,
                   float alpha,
                   DWTCoeffs *coeffs);

#endif /* PHASE_AMP_H */
