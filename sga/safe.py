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

Created on Apr 12, 2017

@author: Matej Ušaj
'''

import math
import sys

import numpy as np
import pandas as p
import scipy.spatial as space
import scipy.stats as stats


class Safe(object):
    enrichmentDF = None
    
    def __init__(self, network_path, attributes_path, neighbors_path=None, distance_threshold=.5, region_specific=False):
        self.network_path = network_path
        self.attributes_path = attributes_path
        
        # Read network file
        if network_path.endswith('.csv'):
            self.network = p.read_csv(network_path, index_col=0)
        else:
            raise Exception("Unknown network file type: %s" % network_path)
        
        # Read attributes file
        if attributes_path.endswith('.csv'):
            self.attributes = p.read_csv(attributes_path, index_col=0)
        else:
            raise Exception("Unknown attributes file type: %s" % attributes_path)
        
        if not neighbors_path:
            self.generate_neighbors(distance_threshold)
    
    def generate_neighbors(self, distance_threshold=.5):
        d = np.percentile(space.distance.pdist(self.network, 'euclidean'), distance_threshold)
        
        dist_mat = p.DataFrame(space.distance_matrix(self.network, self.network)) # pair-wise distances
        dist_mat = dist_mat[dist_mat <= d]
        dist_mat = dist_mat.apply(lambda x: [self.network.index[y] for y in x.dropna().index])
        dist_mat.index = self.network.index
        self.network.loc[:,'neighborhood'] = dist_mat

#     def read_neighbors(self, neighbors_path):
#         self.neighbors = p.read_csv(neighbors_path, header=0, names=['node1', 'node2'])
#         self.neighbors = self.neighbors.groupby('node1')['node2'].apply(lambda x: list(x))
#     
#     def save_neighbors(self, neighbors_path):
#         l = []
#         for node, neighbor_list in self.neighbors.iteritems():
#             l += zip([node] * len(neighbor_list), neighbor_list)
#         p.DataFrame(l, columns=['node1', 'node2']).to_csv(neighbors_path, header=True, index=False)
    
    def prepare_attributes(self):
        self.attributes = self.attributes.reindex(self.network.index.unique(), fill_value=0)
        self.binary_attributes = not np.setdiff1d(np.unique(self.attributes), [0,1])

    def normalizeP(self, p, Fj):
        if p == 0.0: # Some return 0.0 for some reason? Solved as below
            p = sys.float_info.min
        # Convert p-values into normalized neighborhood enrichment scores
        # Min p-value that Matlab can calculate: Pmin = 10**(-16) => -log10(Pmin) = 16.0
        p = min(-math.log10(p), 16.0) / 16.0
        if p < (-math.log10(0.05/len(Fj)) / 16.0): # Significantly enriched attributes
            p = 0.0
        return p

    def hyperwrap(self, U, Sij, N, Fj):
        def hypergeom(A):
            return self.normalizeP(1 - stats.hypergeom.cdf(Sij[A]-1, N, Fj[A], U), Fj)
        return hypergeom
    
    def calculate(self):
        self.enrichmentDF = p.DataFrame(index=self.network.index, columns=self.attributes.columns, dtype=np.float64)
        
        # Number of nodes in the network
        N = len(self.network.index)
        # Number of nodes in the network given any attribute
        Fj = self.attributes.sum().to_dict()
        
        for row in range(len(self.network.index)):
            neighborhood = self.network.iloc[row, 2]
        
            U = len(neighborhood)
            Sij = self.attributes.loc[set(neighborhood)].sum().to_dict()
        
            enrichments = map(self.hyperwrap(U, Sij, N, Fj), self.attributes.columns)
            self.enrichmentDF.iloc[row,:] = enrichments
        
        return self.enrichmentDF
        
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Perform safe analysis on a network')
    parser.add_argument('network', help='Network file. Csv format with header row and 3 columns (name, x, y)')
    parser.add_argument('attributes', help='Attributes file for enrichment test. Csv format with header row and 2+ columns (name, attr1, attr2, ..., attrX)')
    parser.add_argument('output', help='Where to save the results')
    parser.add_argument('neighbors', nargs='?', help='List of neighbors for each node. Csv format with header row and 2 columns (node1, node2)')
    parser.add_argument('-d', '--distance-threshold', dest='distance_threshold',
                       default=0.5,
                       help='Percentile of all pair-wise node distances')
    parser.add_argument('-r', '--region-specific', dest='region_specific', action='store_true')
    
    args = parser.parse_args()
    
    