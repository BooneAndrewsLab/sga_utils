'''
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

Created on Mar 31, 2017

@author: Matej Ušaj
'''

import logging

import numpy as np
import pandas as p


logger = logging.getLogger(__name__)

try:
    from . import c_correlation
    USE_OPT_CORR = True
except ImportError:
    USE_OPT_CORR = False
    logger.warning('Using slower pandas correlation function. Compile ext for an optimized version shipped with this code.')

def correlation(data, axis='rows'):
    if axis not in ('rows', 'columns'):
        raise ValueError('Correlation axis must be either "rows" or "columns".')
    
    if USE_OPT_CORR:
        if axis == 'rows':
            result = p.DataFrame(
                np.zeros((data.shape[0], data.shape[0])), 
                index=data.index, 
                columns=data.index)
            c_correlation.correlation(data, result.values)
        else:
            result = p.DataFrame(
                np.zeros((data.shape[1], data.shape[1])), 
                index=data.columns, 
                columns=data.columns)
            c_correlation.correlation(data.T, result.values)
        return result
    else:
        if axis == 'rows':
            return data.T.corr(min_periods=3) - np.identity(data.shape[0])
        return data.corr(min_periods=3) - np.identity(data.shape[1])
    