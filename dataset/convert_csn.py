#!/usr/bin/env python


# This code is copied from the CodeSearchNet repository (https://github.com/github/CodeSearchNet)
# CodeSearchNet is provided with the MIT License

# MIT License
#
# Copyright (c) 2019 GitHub
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Remove near duplicates from data and perform train/test/va lidation/holdout splits.

todo: python src/dataextraction/dedup_split.py /media/cameron/angelas files/codesearchnet/resources/data/python/final/jsonl/train src/dataextraction/extracted/


Usage:
    dedup_split.py [options] INPUT_FILENAME OUTPUT_FOLDER

Arguments:
    INPUT_FOLDER               directory w/ compressed jsonl files that have a .jsonl.gz a file extension
    OUTPUT_FOLDER              directory where you want to save data to.

Options:
    -h --help                    Show this screen.
    --azure-info=<path>          Azure authentication information file (JSON).
    --train-ratio FLOAT          Ratio of files for training set. [default: 0.6]
    --valid-ratio FLOAT          Ratio of files for validation set. [default: 0.15]
    --test-ratio FLOAT           Ratio of files for test set. [default: 0.15]
    --holdout-ratio FLOAT        Ratio of files for test set. [default: 0.1]
    --debug                      Enable debug routines. [default: False]

Example:

    python dedup_split.py \
    --azure-info /ds/hamel/azure_auth.json \
    azure://semanticcodesearch/pythondata/raw_v2  \
    azure://semanticcodesearch/pythondata/Processed_Data_v2

"""

import hashlib
import pandas as pd

from dpu_utils.utils import RichPath, run_and_debug
from dpu_utils.codeutils.deduplication import DuplicateDetector

from tqdm import tqdm


def jsonl_to_df(input_folder: RichPath) -> pd.DataFrame:
    "Concatenates all jsonl files from path and returns them as a single pandas.DataFrame ."

    assert input_folder.is_dir(), 'Argument supplied must be a directory'
    dfs = []
    files = list(input_folder.iterate_filtered_files_in_dir('*.jsonl.gz'))
    assert files, 'There were no jsonl.gz files in the specified directory.'
    print(f'reading files from {input_folder.path}')
    for f in tqdm(files, total=len(files)):
        dfs.append(pd.DataFrame(list(f.read_as_jsonl(error_handling=lambda m,e: print(f'Error while loading {m} : {e}')))))
    return pd.concat(dfs)


def remove_duplicate_code_df(df: pd.DataFrame) -> pd.DataFrame:
    "Resolve near duplicates based upon code_tokens field in data."
    assert 'code_tokens' in df.columns.values, 'Data must contain field code_tokens'
    assert 'language' in df.columns.values, 'Data must contain field language'
    df.reset_index(inplace=True, drop=True)
    df['doc_id'] = df.index.values
    dd = DuplicateDetector(min_num_tokens_per_document=10)
    filter_mask = df.apply(lambda x: dd.add_file(id=x.doc_id,
                                                 tokens=x.code_tokens,
                                                 language=x.language),
                           axis=1)
    # compute fuzzy duplicates
    exclusion_set = dd.compute_ids_to_exclude()
    # compute pandas.series of type boolean which flags whether or not code should be discarded
    # in order to resolve duplicates (discards all but one in each set of duplicate functions)
    exclusion_mask = df['doc_id'].apply(lambda x: x not in exclusion_set)

    # filter the data
    print(f'Removed {sum(~(filter_mask & exclusion_mask)):,} fuzzy duplicates out of {df.shape[0]:,} rows.')
    return df[filter_mask & exclusion_mask]




def run(args):

    langs = ['Python', 'Javascript']

    for lang in langs:

        input_path = RichPath.create('/home/cameron/PycharmProjects/auto_coding/dataset/csn_data/' + str(lang).lower() + '/final/jsonl/train')
        #input_path = RichPath.create('/media/cameron/angelas files/codesearchnet/resources/data/java/final/jsonl/train')

        # get data and process it
        df = jsonl_to_df(input_path)

        #print('Removing fuzzy duplicates ... this may take some time.')
        # df = remove_duplicate_code_df(df) #todo: this function only works when used on the python part of codesearchnet
        # df = df.sample(frac=1, random_state=None)  # shuffle order of files

        print('Saving data to .pkl file')

        df.to_pickle(lang + '.pkl')


if __name__ == '__main__':
    args = 'docopt(__doc__)'
    print(args)
    run(args=args)
