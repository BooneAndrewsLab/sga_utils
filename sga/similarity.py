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

Created on Mar 21, 2017

@author: Matej Ušaj
'''
from itertools import chain
import logging
import os

import h5py

import numpy as np
import pandas as p

from .toolbox import correlation
from .toolbox.utils import hdf5_read_str_list, read_strain_map
from .toolbox.table_norm import table_normalize


logger = logging.getLogger(__name__)

class Similarity(object):
    '''
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
        
        if isinstance(strain_map, str):
            logger.info("Reading strain map")
            strain_map = read_strain_map(strain_map)
            logger.debug("Found %d strain mappings", len(strain_map))
        
        self.strain_map = strain_map
        
        logger.info("Loading data")
        load_func(input_path1, input_path2)
        logger.info("Data loaded. TS matrix size is %s, FG matrix size is %s", self.ts_data.shape, self.fg_data.shape)
    
    def _bjv_read_scores(self, dataset, root_ele):
        logger.debug("BJV-%s:Reading ORF list", root_ele)
#         orfs = np.array(list(map(lambda x: x.split('_')[1], hdf5_read_str_list(dataset, dataset.get('%s/Cannon/Orf' % (root_ele,))[0]))))
        orfs = np.array(hdf5_read_str_list(dataset, dataset.get('%s/Cannon/Orf' % (root_ele,))[0]))
        
        logger.debug("BJV-%s:Generating query/array indices", root_ele)
        query_idx = np.extract(np.array(dataset.get('%s/Cannon/isQuery' % (root_ele,))[0], dtype=bool), np.arange(*orfs.shape))
        array_idx = np.extract(np.array(dataset.get('%s/Cannon/isArray' % (root_ele,)), dtype=bool).T, np.arange(*orfs.shape))
        
        logger.debug("BJV-%s:Creating DataFrame", root_ele)
        return p.DataFrame(
            dataset.get('%s/eps' % (root_ele,))[array_idx, :][:, query_idx],
            index=orfs[array_idx],
            columns=orfs[query_idx])
    
    def _load_bjv(self, inp, _):
        logger.debug("Reading BJV formatted file %s", inp)
        dataset = h5py.File(inp, 'r')
        
        logger.debug("BJV:Reading TS data")
        self.ts_data = self._bjv_read_scores(dataset, 'ts_merge')
        logger.debug("BJV:Reading FG data")
        self.fg_data = self._bjv_read_scores(dataset, 'fg_merge')
    
    def _load_txt(self, ints, infg):
        raise NotImplementedError('Loading text file is not implemented yet')
    
    def _similarity(self, data):
        logger.info("Input matrix size is %s.", data.shape)
        
        logger.debug("Computing row similarity on %d rows.", data.shape[0])
        corr_rows = correlation.correlation(data, axis='rows')
        
        logger.debug("Computing column similarity on %d columns.", data.shape[1])
        corr_cols = correlation.correlation(data, axis='columns')
        
        if self.strain_map:
            logger.debug("Replacing strain ids with allele names.")
            corr_rows.columns = [self.strain_map[c.split('_')[1]] for c in corr_rows.columns]
            corr_rows.index = [self.strain_map[c.split('_')[1]] for c in corr_rows.index]
            corr_cols.columns = [self.strain_map[c.split('_')[1]] for c in corr_cols.columns]
            corr_cols.index = [self.strain_map[c.split('_')[1]] for c in corr_cols.index]
        
        unified_axis = sorted(set(list(corr_cols.index) + list(corr_rows.index)))
        logger.debug("Unified axis (%d) ready.", len(unified_axis))
        merged = np.array([
            corr_cols.reindex(unified_axis,unified_axis).values, 
            corr_rows.reindex(unified_axis,unified_axis).values])
        
        logger.debug("Combining QQ/AA correlations.")
        return p.DataFrame(np.nanmean(merged, axis=0), index=unified_axis, columns=unified_axis)
    
    def essential_similarity(self):
        logger.info("Computing similarity of essental strains profiles.")
        data = self.ts_data.loc[
            [c for c in self.ts_data.index if '_tsa' in c], 
            [c for c in self.ts_data.columns if '_tsq' in c]]
        self.ts_sim = self._similarity(data)
        return self.ts_sim
    
    def nonessential_similarity(self):
        logger.info("Computing similarity of nonessental strains profiles.")
        data = self.fg_data.loc[
            [c for c in self.fg_data.index if '_dma' in c], 
            [c for c in self.fg_data.columns if '_sn' in c]]
        self.fg_sim = self._similarity(data)
        return self.fg_sim
    
    def similarity(self):
        tsdata = self.ts_data.loc[
            :, 
            [c for c in self.ts_data.columns if ('_y' not in c and '_damp' not in c)]]
        fgdata = self.fg_data.loc[
            :, 
            [c for c in self.fg_data.columns if ('_y' not in c and '_damp' not in c)]]
        
#         if self.strain_map:
#             logger.debug("Replacing strain ids with allele names.")
#             tsdata.columns = [self.strain_map[c] for c in tsdata.columns]
#             tsdata.index = [self.strain_map[c] for c in tsdata.index]
#             fgdata.columns = [self.strain_map[c] for c in fgdata.columns]
#             fgdata.index = [self.strain_map[c] for c in fgdata.index]
#             
#             tsdata = tsdata.iloc[~tsdata.index.duplicated(), ~tsdata.columns.duplicated()]
#             fgdata = fgdata.iloc[~fgdata.index.duplicated(), ~fgdata.columns.duplicated()]
#         
#         unified_axis = sorted(set(chain(tsdata.index, tsdata.columns, fgdata.index, fgdata.columns)))
        
        # Layer 1: FG QQ
        print('fg_qq')
        fg_qq = correlation.correlation(fgdata, axis='columns')

        # Layer 2: TS QQ (normalized based on TS AA)
        print('ts_qq')
        ts_qq = correlation.correlation(tsdata, axis='columns')
        
        common_queries = set(fgdata.columns).intersection(tsdata.columns)
        common_arrays = set(fgdata.index).intersection(tsdata.index)
        
        # cc1 / data1
        print('fg_aa')
        tmp_fg_aa = correlation.correlation(fgdata.reindex(common_arrays, common_queries), axis='rows')
        # cc2 / data2
        print('ts_aa')
        tmp_ts_aa = correlation.correlation(tsdata.reindex(common_arrays, common_queries), axis='rows')
        
        print('normalize')
        ts_qq_norm = table_normalize(tmp_fg_aa, tmp_ts_aa, ts_qq)
        
        # Layer 3: FG AA
        print('fg_aa')
        fg_aa = correlation.correlation(fgdata, axis='rows')
        
        # Layer 4: TS AA (normalized)
        print('ts_aa')
        ts_aa = correlation.correlation(tsdata, axis='rows')
        print('normalize')
        ts_aa_norm = table_normalize(tmp_fg_aa, tmp_ts_aa, ts_aa)
        
        
        
#         return ts_qq_norm
    
    def _save(self, df, path):
        df.to_csv(path, sep='\t', index=True, header=True)

    def save_essential_similarity(self, path=None):
        path = path or os.path.join(self.output, 'cc_ExE.txt')
        logger.info("Saving ExE dataset in %s.", path)
        self._save(self.ts_sim, path)
    
    def save_nonessential_similarity(self, path=None):
        path = path or os.path.join(self.output, 'cc_NxE.txt')
        logger.info("Saving ExE dataset in %s.", path)
        self._save(self.fg_sim, path)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute similarity matrix.')
    parser.add_argument('scores_file', help='First file with SGA scores. See formats for details')
    parser.add_argument('scores_file_2', nargs='?', help='Optional second file with SGA scores. See formats for details')
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
    parser.add_argument('-e', '--skip-essential', dest='exe', action='store_false',
                        help='Do not generate ExE correlations')
    parser.add_argument('-n', '--skip-nonessential', dest='nxn', action='store_false',
                        help='Do not generate NxN correlations')
    parser.add_argument('-a', '--skip-all', dest='all', action='store_false',
                        help='Do not generate ALL correlations')
    
    args = parser.parse_args()
    
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level, format='%(asctime)s\t%(levelname)s:\t%(message)s')
    
    similarity = Similarity(
            args.scores_file,
            args.output_folder,
            args.scores_file_2,
            args.input_format,
            args.strain_map)
    
    if args.exe:
        similarity.essential_similarity()
        similarity.save_essential_similarity()
    
    if args.nxn:
        similarity.nonessential_similarity()
        similarity.save_nonessential_similarity()
    