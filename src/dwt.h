/*
 * WEMA - 2D Discrete Wavelet Transform
 * CDF 9/7 biorthogonal wavelets via lifting scheme
 */

#ifndef DWT_H
#define DWT_H

#include "wema.h"

/*
 * Initialize DWT coefficient storage.
 * Returns 0 on success, -1 on error.
 */
int dwt_init(DWTCoeffs *coeffs, int width, int height, int num_levels);

/*
 * Free DWT coefficient storage.
 */
void dwt_free(DWTCoeffs *coeffs);

/*
 * Forward 2D DWT.
 * Input: grayscale image [height][width]
 * Output: wavelet coefficients
 */
void dwt_forward(const float *image, int width, int height,
                 DWTCoeffs *coeffs);

/*
 * Inverse 2D DWT.
 * Input: (possibly modified) coefficients
 * Output: reconstructed grayscale image
 */
void dwt_inverse(const DWTCoeffs *coeffs, float *image);

/*
 * Get total number of coefficient positions (for buffer sizing).
 */
size_t dwt_num_positions(const DWTCoeffs *coeffs);

/*
 * Compute recommended number of decomposition levels.
 */
int dwt_compute_levels(int width, int height);

#endif /* DWT_H */
