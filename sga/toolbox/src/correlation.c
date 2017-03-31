#include "correlation.h"

#include "math.h"
#include "Python.h"
#include "stdio.h"

void correlation(double* npyArray2D, int npyLength1D, int npyLength2D,
        double* resultArray2D, int resLen1D, int resLen2D) {
    if (npyLength1D != resLen1D || npyLength1D != resLen2D) {
        PyErr_Format(PyExc_ValueError,
                "Output expected (%d,%d) but got (%d,%d)",
                npyLength1D, npyLength1D, resLen1D, resLen2D);
        return;
    }

    int use_flags[npyLength2D];
    int totNaN, totLength, i, j, k;
    double sumX, sumY, sumXY, sumX2, sumY2, den;
    double val1, val2, val;

    for (i = 0; i < npyLength1D; i++) {
        for (j = i + 1; j < npyLength1D; j++) {

            totNaN = sumX = sumY = 0;
            sumXY = sumX2 = sumY2 = 0;

            for (k = 0; k < npyLength2D; k++) {
                val1 = npyArray2D[i * npyLength2D + k];
                val2 = npyArray2D[j * npyLength2D + k];

                if (isnan(val1) || isnan(val2)) {
                    use_flags[k] = 0;
                    totNaN++;
                } else {
                    use_flags[k] = 1;
                    sumX += val1;
                    sumY += val2;
                    sumXY += val1 * val2;
                    sumX2 += val1 * val1;
                    sumY2 += val2 * val2;
                }
            }

            totLength = npyLength2D - totNaN;
            if (totLength < 3) {
                val = NAN;
            } else {  // compute correlation
                den = sqrt(totLength * sumX2 - sumX * sumX)
                        * sqrt(totLength * sumY2 - sumY * sumY);

                if (den < .00001) {
                    val = NAN;
                } else {
                    val = (totLength * sumXY - sumX * sumY) / den;
                }
            }

            resultArray2D[i * resLen1D + j] = val;
            resultArray2D[j * resLen1D + i] = val;
        }
    }
}
