'''
Created on Mar 21, 2017

@author: matej
'''
from itertools import chain
import logging
import os

import h5py

import numpy as np
import pandas as p
from toolbox.utils import read_str_list


class Similarity(object):
    '''
    classdocs
    '''
    
    INPUT_FORMAT_BJV = 'bjv'
    INPUT_FORMAT_TXT = 'txt'

    def __init__(self, input_path1, output_path, input_path2=None, input_format=INPUT_FORMAT_TXT,
                 strain_map=None):
        '''
        Constructor
        '''
        load_func = '_load_%s' % (input_format,)
        if not hasattr(self, load_func):
            raise Exception('Unknown input format "%s"' % (input_format,))
        
        self.output = output_path
        load_func = getattr(self, load_func)
        
        if not os.path.exists(input_path1) or not os.path.isfile(input_path1):
            raise Exception('Input file does not exist "%s"' % (input_path1,))
        
        if input_path2 and (not os.path.exists(input_path2) or not os.path.isfile(input_path2)):
            raise Exception('Input file does not exist "%s"' % (input_path2,))
        
        if input_format == self.INPUT_FORMAT_BJV and not strain_map:
            raise Exception("Ben's format requires strain-allele mapping file")
        
        self.strain_map = strain_map
        
        logging.info("Loading data")
        load_func(input_path1, input_path2)
        logging.info("Data loaded. TS matrix size is %s, FG matrix size is %s", self.ts_data.shape, self.fg_data.shape)
    
    def _bjv_read_scores(self, dataset, root_ele):
        logging.debug("BJV-%s:Reading ORF list", root_ele)
        orfs = np.array(list(map(lambda x: x.split('_')[1], read_str_list(dataset, dataset.get('%s/Cannon/Orf' % (root_ele,))[0]))))
        
        logging.debug("BJV-%s:Generating query/array indices", root_ele)
        query_idx = np.extract(np.array(dataset.get('%s/Cannon/isQuery' % (root_ele,))[0], dtype=bool), np.arange(*orfs.shape))
        array_idx = np.extract(np.array(dataset.get('%s/Cannon/isArray' % (root_ele,)), dtype=bool).T, np.arange(*orfs.shape))
        
        logging.debug("BJV-%s:Creating DataFrame", root_ele)
        return p.DataFrame(
            dataset.get('%s/eps' % (root_ele,))[array_idx, :][:, query_idx],
            index=orfs[array_idx],
            columns=orfs[query_idx])
    
    def _load_bjv(self, inp, _):
        logging.debug("Reading BJV formatted file %s", inp)
        dataset = h5py.File(inp, 'r')
        
        logging.debug("BJV:Reading TS data")
        self.ts_data = self._bjv_read_scores(dataset, 'ts_merge')
        logging.debug("BJV:Reading FG data")
        self.fg_data = self._bjv_read_scores(dataset, 'fg_merge')
    
    def _load_txt(self, ints, infg):
        raise NotImplementedError('Loading text file is not implemented yet')
    
    def _similarity(self, data):
        logging.info("Input matrix size is %s.", data.shape)
        logging.debug("Computing row similarity on %d rows.", data.shape[0])
        corr_rows = data.T.corr(min_periods=3) - np.identity(data.shape[0])
        logging.debug("Computing column similarity on %d columns.", data.shape[1])
        corr_cols = data.corr(min_periods=3) - np.identity(data.shape[1])
        
        if self.strain_map:
            logging.debug("Replacing strain ids with allele names.")
            corr_rows.columns = [self.strain_map[c] for c in corr_rows.columns]
            corr_rows.index = [self.strain_map[c] for c in corr_rows.index]
            corr_cols.columns = [self.strain_map[c] for c in corr_cols.columns]
            corr_cols.index = [self.strain_map[c] for c in corr_cols.index]
        
        unified_axis = sorted(set(list(corr_cols.index) + list(corr_rows.index)))
        logging.debug("Unified axis (%d) ready.", len(unified_axis))
        merged = np.array([
            corr_cols.reindex(unified_axis,unified_axis).values, 
            corr_rows.reindex(unified_axis,unified_axis).values])
        
        logging.debug("Combining QQ/AA correlations.")
        return p.DataFrame(np.nanmean(merged, axis=0), index=unified_axis, columns=unified_axis)
    
    def essential_similarity(self):
        logging.info("Computing similarity of essental strains profiles.")
        data = self.ts_data.loc[
            [c for c in self.ts_data.index if c.startswith('tsa')], 
            [c for c in self.ts_data.columns if c.startswith('tsq')]]
        return self._similarity(data)
    
    def nonessential_similarity(self):
        logging.info("Computing similarity of essental strains profiles.")
        data = self.fg_data.loc[
            [c for c in self.fg_data.index if c.startswith('dma')], 
            [c for c in self.fg_data.columns if c.startswith('sn')]]
        return self._similarity(data)
    
    def quantile_normalization(self, data, refdist):
        ind = data.count().sum()
#         
#         ind = find(~isnan(data));
#         
#         quantiles = 1/length(ind):1/length(ind):1
#         refquant = quantile(refdist,quantiles)
#         
#         [vals,sort_ind]=sort(data(ind),'ascend')
#         norm_dist = data
#         norm_dist(ind(sort_ind))=refquant
    
    def similarity(self):
        tsdata = self.ts_data.loc[
            :, 
            [c for c in self.ts_data.columns if not c.startswith('y') and not c.startswith('damp')]]
        fgdata = self.fg_data.loc[
            :, 
            [c for c in self.fg_data.columns if not c.startswith('y') and not c.startswith('damp')]]
        
        if self.strain_map:
            logging.debug("Replacing strain ids with allele names.")
            tsdata.columns = [strain_map[c] for c in tsdata.columns]
            tsdata.index = [strain_map[c] for c in tsdata.index]
            fgdata.columns = [strain_map[c] for c in fgdata.columns]
            fgdata.index = [strain_map[c] for c in fgdata.index]
        
        unified_axis = sorted(set(chain(tsdata.index, tsdata.columns, fgdata.index, fgdata.columns)))
        
        # Layer 1: FG QQ
        fg_qq = fgdata.corr(min_periods=3) - np.identity(fgdata.shape[1])
        
        # Layer 2: TS QQ (normalized based on TS AA)
        ts_qq = tsdata.corr(min_periods=3) - np.identity(tsdata.shape[1])

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute similarity matrix.')
    parser.add_argument('scores_file', help='First file with SGA scores. See formats for details')
    parser.add_argument('scores_file_2', nargs='?', help='Second file with SGA scores. See formats for details')
    parser.add_argument('output_folder', help='Where to save the results')
    parser.add_argument('-f', '--input-format', dest='input_format',
                       choices=[Similarity.INPUT_FORMAT_TXT, Similarity.INPUT_FORMAT_BJV],
                       default=Similarity.INPUT_FORMAT_TXT,
                       help="""Format of the input files. If format is '%s' then the first file is TS dataset and second file is FG.
If format is '%s' then first file is the .mat workspace snapshot and the second input file should be omitted.""" % (Similarity.INPUT_FORMAT_TXT, Similarity.INPUT_FORMAT_BJV))
    parser.add_argument('-m', '--strain-map', dest='strain_map',
                        help='Strain -> allele mapping file. Used if axis labels are strains and not alleles.'
                        'This file should have two tab separated columns and no header. Column A is strain id, column B is allele name. '
                        'ie: "tsq123{tab}bla1-123". '
                        'This option is mandatory for Ben\'s format.')
    parser.add_argument('-l', '--log', dest='loglevel', default='INFO',
                       help='Log level to use')
    
    args = parser.parse_args()
    
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level, format='%(asctime)s\t%(levelname)s:%(message)s')
    
    strain_map = None
    if args.strain_map:
        logging.info("Reading strain map")
        with open(args.strain_map) as sm:
            strain_map = {tuple(l.strip().split('\t')) for l in sm}
        logging.debug("Found %d strain mappings", len(strain_map))
    
    similarity = Similarity(
            args.scores_file,
            args.output_file,
            args.scores_file_2,
            args.input_format,
            strain_map)
    