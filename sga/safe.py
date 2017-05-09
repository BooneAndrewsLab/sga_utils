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

import six

import numpy as np
import pandas as p
import scipy.spatial as space
import scipy.stats as stats


class Safe(object):
    enrichmentDF = None
    
    def __init__(self, network, attributes, neighbors=None, distance_threshold=.5, region_specific=False):
        # Read network file
        if isinstance(network, six.string_types):
            if network.endswith('.csv'):
                self.network = p.read_csv(network, index_col=0)
            else:
                raise Exception("Unknown network file type: %s" % network)
        else:
            self.network = network
        
        # Read attributes file
        if isinstance(attributes, six.string_types):
            if attributes.endswith('.csv'):
                self.attributes = p.read_csv(attributes, index_col=0)
            else:
                raise Exception("Unknown attributes file type: %s" % attributes)
        else:
            self.attributes = attributes
        
        if not neighbors:
            self.generate_neighbors(distance_threshold)
        elif isinstance(neighbors, six.string_types):
            self.read_neighbors(neighbors)
        else:
            self.network = self.network.join(neighbors)
    
    def generate_neighbors(self, distance_threshold=.5):
        d = np.percentile(space.distance.pdist(self.network, 'euclidean'), distance_threshold)
        
        dist_mat = p.DataFrame(space.distance_matrix(self.network, self.network)) # pair-wise distances
        dist_mat = dist_mat[dist_mat <= d]
        dist_mat = dist_mat.apply(lambda x: [self.network.index[y] for y in x.dropna().index])
        dist_mat.index = self.network.index
        self.network.loc[:,'neighborhood'] = dist_mat

    def read_neighbors(self, neighbors_path):
        neighbors = p.read_csv(neighbors_path, header=0, names=['node1', 'neighborhood'])
        self.network = self.network.join(neighbors.groupby('node1')['neighborhood'].apply(list))
#     
    def save_neighbors(self, neighbors_path):
        l = []
        for node, neighbor_list in self.network.iloc[:,-1:].itertuples():
            l += zip([node] * len(neighbor_list), neighbor_list)
        p.DataFrame(l, columns=['node1', 'neighborhood']).to_csv(neighbors_path, header=True, index=False)
    
    def prepare_attributes(self):
        self.attributes = self.attributes.reindex(self.network.index.unique(), fill_value=0)
        self.binary_attributes = not np.setdiff1d(np.unique(self.attributes), [0,1])

    def normalizeP(self, p, Fj):
#         if p == 0.0: # Some return 0.0 for some reason? Solved as below
#             p = sys.float_info.min
        # Convert p-values into normalized neighborhood enrichment scores
        # Min p-value that Matlab can calculate: Pmin = 10**(-16) => -log10(Pmin) = 16.0
        p = min(-np.log10(p), 16.0) / 16.0
        if p < (-np.log10(0.05/len(Fj)) / 16.0): # Significantly enriched attributes
            p = 0.0
        return p

    def hyperwrap(self, U, Sij, N, Fj):
        def hypergeom(A):
            return self.normalizeP(1 - stats.hypergeom.cdf(Sij[A]-1, N, Fj[A], U), Fj)
        return hypergeom
    
    def calculate(self, force_python_impl=False):
        from .toolbox import USE_C_OPT
        
        # TODO: compare cdf outputs of c and python implementation
        
        if USE_C_OPT and not force_python_impl:
            from .toolbox import c_impl
            
            enrichment = np.zeros((len(self.network.index), len(self.attributes.columns)))
            
            cdata = self.network.iloc[:,2].apply(lambda x: self.attributes.ix[set(x)].sum())
            cdata['len'] = self.network.iloc[:,2].apply(len)
            cdata = cdata.astype(np.uint32)
            
            Fj = self.attributes.sum().astype(np.uint32)
            
            c_impl.safe(cdata, enrichment, Fj)
            
            self.enrichmentDF = p.DataFrame(enrichment, index=self.network.index, columns=self.attributes.columns)
        else:
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
                self.enrichmentDF.iloc[row,:] = list(enrichments)
            
        return self.enrichmentDF
    
    def plot(self):
        from matplotlib import pyplot as plt
        def centeroidnp(arr):
            length = arr.shape[0]
            sum_x = np.sum(arr.iloc[:, 0])
            sum_y = np.sum(arr.iloc[:, 1])
            return sum_x/length, sum_y/length
        
        cx, cy = centeroidnp(self.network[['x', 'y']])
        r = self.network.apply(lambda x: np.sqrt(np.power(cx - x['x'], 2) + np.power(cy - x.y, 2)), axis=1).max()
        
        an = np.linspace(0, 2*np.pi, 100)
        
        fig = plt.figure()
        plt.plot(r*np.cos(an) + cx, r*np.sin(an) - cy, 'w--')
        _ = plt.axis('equal')
        _ = plt.axis('off')
        fig.patch.set_facecolor('black')
        fig.set_size_inches((20,20))
        
        for term in self.enrichmentDF.columns:
            base_line = None
            for node, enr in self.enrichmentDF[term][self.enrichmentDF[term] > 0].iteritems():
                node_data = self.network.ix[node]
                base_line, = plt.plot(node_data.x, -node_data.y, 'o', alpha=enr, mew=0, color=base_line and base_line.get_color())
        
        return fig
    
def main():
    import argparse, os
    
    parser = argparse.ArgumentParser(description='Perform safe analysis on a network')
    parser.add_argument('network', help='Network file. Csv format with header row and 3 columns (name, x, y)')
    parser.add_argument('attributes', help='Attributes file for enrichment test. Csv format with header row and 2+ columns (name, attr1, attr2, ..., attrX)')
    parser.add_argument('output', help='Where to save the results')
    parser.add_argument('neighbors', nargs='?', help='List of neighbors for each node. Csv format with header row and 2 columns (node1, node2)')
    parser.add_argument('-d', '--distance-threshold', dest='distance_threshold',
                       default=0.5,
                       help='Percentile of all pair-wise node distances')
    parser.add_argument('-r', '--region-specific', dest='region_specific', action='store_true')
    parser.add_argument('-p', '--save-plot', dest='save_plot', action='store_true')
    
    args = parser.parse_args()
    
    safe = Safe(args.network, args.attributes, args.neighbors, args.distance_threshold, args.region_specific)
    safe.prepare_attributes()
    enr = safe.calculate()
    
    enr.to_csv(args.output)
    
    if args.save_plot:
        base, _ext = os.path.splitext(args.output)
        
        fig = safe.plot()
        fig.savefig(base + '.jpg', bbox_inches='tight', facecolor='black')
    