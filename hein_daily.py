import os
import re
import numpy as np
import pandas as pd
from datetime import datetime

class DatasetHeinDaily:

    def __init__(self, directory, drop_independent=True, keep_missing_speakerid=True):
        self.directory = directory
        
        self.descr:dict = get_file_names('descr', self.directory) #id : filename
        self.speeches:dict = get_file_names('speeches', self.directory)
        self.speaker_map:dict = get_file_names('SpeakerMap', self.directory)
        assert not set(self.descr.keys()).symmetric_difference(self.speeches.keys()) and not set(self.descr.keys()).symmetric_difference(self.speaker_map.keys()), 'The ids of the files do not overlap'
        
        self.drop_independent = drop_independent
        self.keep_missing_speakerid = keep_missing_speakerid #TODO

        self.track_sizes = pd.DataFrame(columns=['speeches', 'after_merge', 'after_drop_independent']) # for verbosity

    def preprocess(self):

        pd_to_concat = []
        for i in self.descr.keys():
            S = pd.read_csv(os.path.join(self.directory, self.speeches[i]), sep='|', encoding='latin_1', on_bad_lines='skip')
            D = pd.read_csv(os.path.join(self.directory, self.descr[i]), sep='|', encoding='latin_1', \
                             parse_dates=['date'], date_parser=lambda x: datetime.strptime(x, '%Y%m%d'))
            SM = pd.read_csv(os.path.join(self.directory, self.speaker_map[i]), sep='|', encoding='latin_1')
            
            self.track_sizes.loc[i, 'speeches'] = len(S)
            
            X = S.merge(D, on='speech_id').merge(SM, on='speech_id')
            self.track_sizes.loc[i, 'after_merge'] = len(X)

            if self.drop_independent:
                X = X[X['party']!='I']
                self.track_sizes.loc[i, 'after_drop_independent'] = len(X)

            pd_to_concat.append(X)

        return pd.concat(pd_to_concat, axis=0, ignore_index=True)
            
def get_file_names(file_category, directory)->dict:
    # ex: file_category = 'speeches'
    out = {} # digit_id: file_name  (speeches_110.txt -> digit_id = 110 )
    for file_name in os.listdir(directory):
        rgx = r'^.*%s.*$'%(file_category)
        if re.search(rgx, file_name):
            id = re.findall(r'\d+', file_name)
            assert len(id) == 1, f'Ambigious/No digit IDs found for {file_name}. Found: {id}'
            assert id[0] not in out, f'{id[0]} exists multiple times'
            out[id[0]] = file_name
    return out

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='resources/hein-daily')
    parser.add_argument("--target_dir", type=str, default='.')
    args = parser.parse_args()

    ds = DatasetHeinDaily(directory=args.input_dir)
    print('Preprocessing hein daily data...')
    dataset = ds.preprocess()

    print('\n Sanity check number of Speeches after certain steps in process:')
    print('Unprocessed = %i'%(ds.track_sizes.sum(axis=0)['speeches']) )
    print('After merge = %i'%(ds.track_sizes.sum(axis=0)['after_merge']) )
    print('After drop independent = %i'%(ds.track_sizes.sum(axis=0)['after_drop_independent']) )

    print(f'Final shape of dataset = {np.shape(dataset)}')
    print()
    
    path = os.path.join(args.target_dir, 'hein_daily.csv')
    dataset.to_csv(path, index=False)
    print('Saved to %s.'%(path))

