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

Created on Apr 10, 2017

@author: Matej Ušaj
'''

import logging

import numpy as np
import pandas as p

from . import USE_C_OPT


logger = logging.getLogger(__name__)

def _c_normalize(data3_tableix, t1, data3_nn, cpu=1):
    from . import c_impl
    c_impl.table_norm(data3_nn, t1, data3_tableix)

def _normalize(index, t1, data3_nn):
    for x in t1:
        index += data3_nn > x
    
    index[index == 0] = 1

def normalize(table, t1, t2):
    data = table.values.flatten()
    
    result = np.full_like(data, np.nan)
    data_values = ~np.isnan(data)
    
    data = data[data_values]
    data_index = np.zeros_like(data, dtype=np.int64)
    
    if USE_C_OPT:
        _c_normalize(data_index, t1, data)
    else:
        _normalize(data_index, t1, data)
    
    data_index -= 1 # leftover from matlab code conversion
    
    result[data_values] = t2[data_index]
    
    return p.DataFrame(result.reshape(table.shape), index=table.index, columns=table.columns)

def _quantile_normalize(data, refdist):
    percentiles = np.linspace(100. / data.shape[0], 100, num=data.shape[0])
    ref_quantiles = np.percentile(refdist, percentiles, interpolation='midpoint') # interpolation used in matlab
    sort_ind = np.argsort(data, kind='mergesort') # sorting alg used in matlab
    result = np.zeros_like(data)
    result[sort_ind] = ref_quantiles
    return result

def table_normalize(data1, data2, data3):
    data1 = data1.values.flatten()
    data2 = data2.values.flatten()
    
    nn = ~np.isnan(data1) & ~np.isnan(data2) # extract cells with values in both arrays
    data2_norm = np.full_like(data2, np.nan)
    data2_norm[nn] = _quantile_normalize(data2[nn], data1[nn]);
    
    table = p.DataFrame({'data2': data2[nn], 'data2_norm': data2_norm[nn]})
    table = table.sort_values('data2', kind='mergesort')
    table = table.groupby('data2').median().reset_index()
    
    return normalize(data3, table.data2, table.data2_norm)
