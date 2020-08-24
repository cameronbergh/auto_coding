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

from transformers import GPT2Tokenizer

from model import GPTSingleHead
from multiprocessing import Pool, Process, Manager, Queue
import random
import time

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def divide_list(seq, num):
    #from Max Shawabkeh
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    # i dont need this function anymore but i like it too much to delete it.
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def process_files(filetypes_to_use, tokenizer, in_q, out_q ):

    finished = 0
    while not in_q.empty():

        path = in_q.get()

        # convert the string to a RichPath and then get the file extension in lowercase.
        parsed = str(pathlib.Path(str(path)).suffix).lower()
        file_ext = parsed[1:]

        # check if this file is on our list of files to use
        if file_ext in filetypes_to_use:
            try:
                # open each file and tokenize it
                with open(path, "r") as f:

                    code_content = f.read()
                    code_content = tokenizer.encode(code_content)

                    # if file is too small, skip it
                    if len(code_content) < 3:
                        print('>>>SKIPPING STUB FILE length: ' + str(len(code_content)))
                        print(code_content)
                        print('<<< ' + str(path))
                    else:
                        # add it to dataframes
                        frame_dict = {'code': code_content, 'language': file_ext}
                        out_q.put(frame_dict)

            # todo: this is bad error handling
            except (RuntimeError, TypeError, NameError):
                print('error')
                continue


def run(linux_path, filetypes_to_use, num_workers, model_path):

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    # get all filenames and paths in the linux kernel dir
    files = []
    for root, d_names, f_names in os.walk(linux_path):
        for f in f_names:
            files.append(os.path.join(root, f))

    file_count = len(files)

    print('found ' + str(file_count) + ' files in the linux folder')

    # init our queue for the procs to put stuff in
    in_q = Queue()
    out_q = Queue()

    #shuffle the list because if we dont then some threads finish too early.
    random.shuffle(files)

    #put files into work queue
    for item in files:
        in_q.put(item)

    print(len(files))
    # del files

    #make processes to tokenize the files
    jobs = []
    for i in range(0, num_workers):
        p = Process(target=process_files, args=(filetypes_to_use, tokenizer, in_q, out_q))
        jobs.append(p)
        p.start()
        print('starting thread ' + str(i))

    print('waiting for threads to finish')

    pbar = tqdm(total=file_count)

    #pause the program until all processes are finished

    tokenized_list = []

    while True:
        try:
            code_content = out_q.get(block=False)
            dictionary = {'code': code_content['code'], 'language': code_content['language']}
            tokenized_list.append(dictionary)
        except:
            pbar.n = out_q.qsize()
            pbar.refresh()
            #print('files tokenized: ' + str(len(tokenized_list)))
            time.sleep(1)

        #if both queues are empty we are almost done!
        if in_q.empty() and out_q.empty():
            print('both queues empty')
            break


    print('making list into pandas dataframe')
    df = pd.DataFrame(tokenized_list, columns=['code', 'language'])

    # df.to_pickle('linux.pkl')
    print(len(df))

    # print('Removing fuzzy duplicates ... this may take some time.')
    # df = remove_duplicate_code_df(df)
    # df = df.sample(frac=1, random_state=None)  # shuffle order of files

    for item in filetypes_to_use:
        print('locating ' + item)
        new_df = df.loc[df['language'].isin([item])]
        new_df.reset_index(drop=True, inplace=True)
        print(new_df)
        print('saving ' + item + '.pkl')
        new_df.to_pickle('dataset/' + item + '.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--linux_dir', type=str, default='dataset/csnl_data/linux-5.9-rc2/', help='path to linux sources')
    args = parser.parse_args()
    linux_path = args.linux_dir


    filetypes_to_use = ['sh', 'c', 'h']

    num_workers = 1 # this must remain one due to a race-condition #todo: fix it

    model_path = 'model/distilgpt2_fine_tuned_coder/'

    run(linux_path, filetypes_to_use, num_workers, model_path)
