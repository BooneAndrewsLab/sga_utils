'''
Created on Mar 22, 2017

@author: matej
'''

def read_str_list(hdf5file, listref):
    return [''.join([chr(z) for z in hdf5file[y][:]]) for y in listref]
