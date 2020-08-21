#!/usr/bin/env python

"""

    # This code is copied and modified from the CodeSearchNet repository (https://github.com/github/CodeSearchNet)
    # CodeSearchNet is provided with the MIT License

    this program, is to be used after 'download_csn.py' to convert the downloaded files into pandas files,
    this is done because pandas is fast and i hope to eventually use dask instead.

Usage:
    download_csn.py --dir DESTINATION_DIR

    example:
        "python download_csn.py"
        "python download_csn.py --dir ./csn_data/"


Options:
    -h --help   Show this screen.

"""

import hashlib
import pandas as pd
from tqdm import tqdm
from dpu_utils.utils import RichPath, run_and_debug
from dpu_utils.codeutils.deduplication import DuplicateDetector
import os
import argparse


def jsonl_to_df(input_folder: RichPath) -> pd.DataFrame:
    "Concatenates all jsonl files from path and returns them as a single pandas.DataFrame ."

    assert input_folder.is_dir(), 'Argument supplied must be a directory'

    dfs = []

    files = list(input_folder.iterate_filtered_files_in_dir('*.jsonl.gz'))

    assert files, 'There were no jsonl.gz files in the specified directory.'

    print(f'reading files from {input_folder}')
    for f in tqdm(files, total=len(files)):

        #read jsonl file into pandas dataframe
        df = pd.DataFrame(list(f.read_as_jsonl(error_handling=lambda m,e: print(f'Error while loading {m} : {e}'))))

        #remove all data that isnt 'code' or 'langauge'
        # note: codesearchnet has the code pre-tokenized, which may be useful but here i discard it along
        # with anything that isnt 'code' or 'langauge'. I do this is because i dont know much about tokenizers
        # or pytorch and since we are doing the tokenizing on the fly in the dataset object,
        # i want to make sure its done correctly. #todo: determine if this is the best way to do this.
        df = df.drop(columns=['repo', 'path', 'func_name', 'original_string', 'code_tokens', 'docstring', 'docstring_tokens', 'sha', 'url', 'partition'])

        dfs.append(df)
    return pd.concat(dfs)


#todo: this function only works when used on the python part of codesearchnet so figure out how to make that work.
#       the duplication seems to be minimal though, so im going to just keep going
"""
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
"""


def run(csn_root_dir):
    langs = ['Python', 'Javascript', 'Java', 'Php', 'Ruby', 'Go']

    for lang in langs:

        lang = lang.lower()

        #todo: there are actually some files in folders other than 'train', and also, i dont like this code.
        input_path = os.path.join(csn_root_dir, str(lang).lower(), 'final', 'jsonl', 'train')
        input_path = os.path.abspath(input_path)
        input_path = RichPath.create(str(input_path))

        print('reaning jsonl files from ' + str(input_path))

        # get data and process it
        df = jsonl_to_df(input_path)

        #print('Removing fuzzy duplicates ... this may take some time.')
        # df = remove_duplicate_code_df(df)
        # df = df.sample(frac=1, random_state=None)  # shuffle order of files

        print('Saving data to .pkl file')
        df.to_pickle(lang + '.pkl')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--csn_dir', type=str, default='csn_data/',
                        help='the length of each example')
    args = parser.parse_args()
    print(args)
    run(csn_root_dir=args.csn_dir)
