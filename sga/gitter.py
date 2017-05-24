'''
MIT License

Copyright (c) 2017 Matej Usaj

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

Created on May 17, 2017

@author: Matej Usaj
'''
from datetime import datetime
import logging
import os

from PIL import Image
import peakutils
import six
from skimage.io import imread
from skimage.transform import radon, rescale, rotate

import numpy as np


PLATE_FORMATS = {
    1536: (32, 48),
    768:  (32, 48), 
    384:  (16, 24),
    96:    (8, 12)
}

logger = logging.getLogger(__name__)

class Gitter():
    def __init__(self):
        pass
    
    def autorotate(self, im, eps=5, steps_in_degree=5):
        thumb = rescale(im, scale=1.* 500 / im.shape[1], mode='reflect')
        thumb = thumb[:min(thumb.shape), :min(thumb.shape)]
        
        theta = np.linspace(-eps, eps, eps * 2 * steps_in_degree + 1)
        sinogram = radon(thumb, theta=theta, circle=False)
        v = sinogram.var(axis=0)
        peaks = peakutils.indexes(v, min_dist=1./steps_in_degree)
        
        if len(peaks) == 1:
            a = peaks[0]
        else:
            mid = eps * steps_in_degree
            closest = peaks[np.absolute(peaks - mid).argmin()]
            peaks = [p for p in peaks if p != closest]
            a = peaks[v[peaks].argmax()]
        
        a = -eps + (a * (1. / steps_in_degree))
        
        return rotate(im, -a)
    
    def process_image(self, images, plate_format=(32, 48), remove_noise=False,
                      autorotate=False, inverse=False, image_align=True,
                      contrast=None, fast=None, grid_save=True, dat_save=True,
                      fx=2.0, is_ref=False, params=None):
        if not isinstance(plate_format, tuple):
            raise ValueError("Unknown plate format %s" % (plate_format, ))
        
        if fast and (fast < 1500 or fast > 4000):
            raise ValueError('Fast resize width must be between 1500-4000px')
        
        expf = 1.5
        nrow, ncol = plate_format
        ptm = datetime.now()
        
        for image in images:
            logger.info("Reading image %s", image)
            
            im = imread(image, as_grey=True)
            
            if fast:
                im = rescale(im, scale=1. * fast / im.shape[1], mode='reflect')
                logger.info("Image resized")
            
            if autorotate:
                im = self.autorotate(im)
                logger.info("Image rotated")
            
            if contrast:
                
                logger.info("Contrast set")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform safe analysis on a network')
    parser.add_argument('network', help='Network file. Csv format with header row and 3 columns (name, x, y)')
    
    args = parser.parse_args()
    