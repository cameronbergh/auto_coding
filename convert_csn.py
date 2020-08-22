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
from model import GPTSingleHead

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

# we load up the model so we can use the tokenizer
model = GPTSingleHead('distilgpt2', max_seq_length=256)
model.add_special_words({"pad_token": "<pad>",
                         "additional_special_tokens": ["<python>", "<javascript>", "<java>", "<php>", "<ruby>",
                                                       "<go>", "<c>", "<h>", "<sh>"]})

def tokenize_jsonl(input_folder: RichPath) -> pd.DataFrame:
    "Concatenates all jsonl files from path and returns them as a single pandas.DataFrame ."

    assert input_folder.is_dir(), 'Argument supplied must be a directory'

    #init list of code samples
    tokenized_list = []

    #check for list of files to read
    files = list(input_folder.iterate_filtered_files_in_dir('*.jsonl.gz'))
    assert files, 'There were no jsonl.gz files in the specified directory.'

    print(f'reading files from {input_folder}')
    for f in tqdm(files, total=len(files)):

        print('tokenizing ' + str(f))
        list_of_code = list(f.read_as_jsonl(error_handling=lambda m,e: print(f'Error while loading {m} : {e}')))

        for item in tqdm(list_of_code):

            #tokenize the code
            tokenized_code = model.tokenizer.encode(item['code'])

            #skip short files, i dont think this ever happens in the csn dataset
            if len(tokenized_code) < 3:
                print('skipping stub file')
                continue

            tokenized_dict = {'code': tokenized_code, 'language': item['language']}
            tokenized_list.append(tokenized_dict)

    return tokenized_list

def run(csn_root_dir):

    langs = ['Python', 'Javascript', 'Java', 'Php', 'Ruby', 'Go']

    for lang in langs:

        lang = lang.lower()

        #todo: there are actually some files in folders other than 'train', and also, i dont like this block of code.
        input_path = os.path.join(csn_root_dir, str(lang).lower(), 'final', 'jsonl', 'train')
        input_path = os.path.abspath(input_path)
        input_path = RichPath.create(str(input_path))

        print('reaning jsonl files from ' + str(input_path))

        # make the jsonl files into a list of dicts
        tokenized_list = tokenize_jsonl(input_path)

        # make the list of dicts into a pandas dataframe for fast saving and loading.
        df = pd.DataFrame(tokenized_list, columns=['code', 'language'])

        print(df)

        #print('Removing fuzzy duplicates ... this may take some time.')
        # df = remove_duplicate_code_df(df)
        # df = df.sample(frac=1, random_state=None)  # shuffle order of files

        print('Saving data to .pkl file')
        df.to_pickle(lang + '.pkl')
        del df


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--csn_dir', type=str, default='dataset/csn_data/',
                        help='the length of each example')
    args = parser.parse_args()
    print(args)
    run(csn_root_dir=args.csn_dir)
