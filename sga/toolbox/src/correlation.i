%module c_correlation

// Numpy Related Includes:
%{
#define SWIG_FILE_WITH_INIT
%}
// numpy arrays
%include "numpy.i"
%init %{
import_array(); // This is essential. We will get a crash in Python without it.
%}
// These names must exactly match the function declaration.
%apply (double* IN_ARRAY2, int DIM1, int DIM2) \
      {(double* npyArray2D, int npyLength1D, int npyLength2D)}
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) \
      {(double* resultArray2D, int resLen1D, int resLen2D)}

%include "correlation.h"

%clear (double* npyArray2D, int npyLength1D, int npyLength2D);
%clear (double* resultArray2D, int resLen1D, int resLen2D);