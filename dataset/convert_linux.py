#!/usr/bin/env python

"""
    # some of This code might be copied and modified from the CodeSearchNet repository (https://github.com/github/CodeSearchNet)
    # CodeSearchNet is provided with the MIT License

    #todo: doc string
    #todo: parallelize everything

    #todo: i think we could also do makefiles and i think that including the filename/path would help those and probably all of them

"""

import hashlib
import pandas as pd
from tqdm import tqdm
from dpu_utils.utils import RichPath, run_and_debug
from dpu_utils.codeutils.deduplication import DuplicateDetector
import os
import argparse
import pathlib

def run():
    import os

    #input the location of your linux kernel sources.
    path = "linux-5.9-rc1/"

    #get all filenames and paths in the linux kernel dir
    files = []
    for root,d_names,f_names in os.walk(path):
        for f in f_names:
            files.append(os.path.join(root, f))

    print('found ' + str(len(files)) + ' files in the linux folder')

    dfs = []

    for path in tqdm(files):
        parsed = str(pathlib.Path(str(path)).suffix).lower() # todo: epic one-liner
        file_ext = parsed[1:]

        try:
            with open(path, "r") as f:
                code_content = f.read()

                if len(code_content) < 32:
                    print('>>>>>>>>>>>>>>>>> SKIPPING STUB FILE length: ' + str(len(code_content)))
                    print(code_content)
                    print('<<<<<<<<<<<<<<<<< ' + str(path))
                else:
                    frame_dict = {'code': [code_content], 'language': [file_ext]}
                    df = pd.DataFrame(frame_dict, columns=['code', 'language'])
                    dfs.append(df)
        except:
            #todo: this is bad error handling
            print('skipping file because non utf8 char ' + str(path))
            continue

    df = pd.concat(dfs)
    print(df)

    for item in ['c', 'h', 'pl', 'sh', 's']:

        # .S files i think is assembly files and i dont even know how to look at that so im leaving it out for now

        print('locating ' + item)
        new_df = df.loc[df['language'].isin([item])]
        print('saving ' + item + '.pkl')

        new_df.to_pickle(item + '.pkl')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--csn_dir', type=str, default='csn_data/', help='the length of each example')
    args = parser.parse_args()
    print(args)
    run()
