/*
 * WEMA - Dual-Tree Complex Wavelet Transform
 */

#ifndef DTCWT_H
#define DTCWT_H

#include "wema.h"

/*
 * Initialize DT-CWT coefficient storage.
 * Returns 0 on success, -1 on error.
 */
int dtcwt_init(DTCWTCoeffs *coeffs, int width, int height, int num_levels);

/*
 * Free DT-CWT coefficient storage.
 */
void dtcwt_free(DTCWTCoeffs *coeffs);

/*
 * Forward 2D DT-CWT.
 * Input: grayscale image [height][width]
 * Output: complex wavelet coefficients
 */
void dtcwt_forward(const float *image, int width, int height,
                   DTCWTCoeffs *coeffs);

/*
 * Inverse 2D DT-CWT.
 * Input: (possibly modified) complex coefficients
 * Output: reconstructed grayscale image
 */
void dtcwt_inverse(const DTCWTCoeffs *coeffs, float *image);

/*
 * Get total number of coefficient positions (for phase buffer sizing).
 */
size_t dtcwt_num_positions(const DTCWTCoeffs *coeffs);

/*
 * Compute recommended number of decomposition levels.
 */
int dtcwt_compute_levels(int width, int height);

#endif /* DTCWT_H */
