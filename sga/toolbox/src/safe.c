/**
MIT License

Copyright (c) 2017 Matej Ušaj

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <gsl/gsl_cdf.h>

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

void safe(unsigned int* data, int dat1d, int dat2d,
          double* enrichment, int enr1d, int enr2d,
          unsigned int* Fj, int Fj1d) {
    const int N = dat1d;
//     const double eps = -log10(.05/Fj1d) / 16.;

    unsigned int U;
    int i, j, datIdx;
    double cdf, normcdf;

    for (i = 0; i < dat1d; i++) {
        U = data[(i + 1) * dat2d - 1]; // last element of current row is the length

        for (j = 0; j < enr2d; j++) {
            datIdx = i * dat2d + j;
            if (data[datIdx] == 0) continue;

            cdf = 1 - gsl_cdf_hypergeometric_P(data[datIdx] - 1, Fj[j], N - Fj[j], U);

            normcdf = -log10(cdf);
            normcdf = isinf(normcdf) ? 16. : normcdf;

            normcdf = min(normcdf, 16.) / 16.;
//             if (normcdf < eps) normcdf = 0.;

            enrichment[i * enr2d + j] = normcdf;
        }
    }
}
