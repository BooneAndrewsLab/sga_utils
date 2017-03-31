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

Created on Mar 22, 2017

@author: Matej Ušaj
'''

def hdf5_read_str_list(hdf5file, listref):
    """Reads a list of strings from hdf5 file

    Read a list of strings

    Args:
        hdf5file: h5py dataset file.
        listref: reference to string list in the hdf5 file.

    Returns:
        List of strings converted from the hdf5 format
    """
    return [''.join([chr(z) for z in hdf5file[y][:]]) for y in listref]

def read_strain_map(path):
    """Read a standard format of strain map.

    Construct a dictionary of strain -> allele pairs.

    Args:
        path: Path to the file with mappings

    Returns:
        A dict mapping strain id's to allele names. For example:

        {'tsq123': 'sla1-23',
         'sn234': 'act1'}
    """
    with open(path) as sm:
        return dict(tuple(l.strip().split('\t')) for l in sm)
