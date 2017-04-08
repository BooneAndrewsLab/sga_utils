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