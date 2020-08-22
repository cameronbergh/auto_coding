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
from model import GPTSingleHead
from multiprocessing import Pool, Process, Manager, Queue
import random
import time



import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)




def divide_list(seq, num):
    #from Max Shawabkeh
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def process_files(filetypes_to_use, model, in_q, out_q):

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
                    code_content = model.tokenizer.encode(code_content)

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

def run(path, filetypes_to_use, num_workers):

    #load model to use its tokenizer #todo: there must be a better way
    model = GPTSingleHead('distilgpt2', max_seq_length=256)
    model.add_special_words({"pad_token": "<pad>",
                             "additional_special_tokens": ["<python>", "<javascript>", "<java>", "<php>", "<ruby>",
                                                           "<go>", "<c>", "<h>", "<sh>"]})

    # get all filenames and paths in the linux kernel dir
    files = []
    for root, d_names, f_names in os.walk(path):
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
    del files

    #make processes to tokenize the files
    jobs = []
    for i in range(0, 8):
        p = Process(target=process_files, args=(filetypes_to_use, model, in_q, out_q))
        jobs.append(p)
        p.start()
        print('starting thread ' + str(i))

    print('waiting for threads to finish')

    pbar = tqdm(total=file_count)




    #pause the program until all processes are finished
    while not in_q.empty():
        time.sleep(.1)
        pbar.n = out_q.qsize()
        pbar.refresh()

    pbar.close()

    print('files tokenized: ' + str(out_q.qsize()))
    tokenized_list = []
    while not out_q.empty():
        tokenized_list.append(out_q.get(block=True, timeout=1))

    df = pd.DataFrame(tokenized_list, columns=['code', 'language'])

    print(df)

    # print('Removing fuzzy duplicates ... this may take some time.')
    # df = remove_duplicate_code_df(df)
    # df = df.sample(frac=1, random_state=None)  # shuffle order of files

    for item in filetypes_to_use:
        print('locating ' + item)
        new_df = df.loc[df['language'].isin([item])]
        print('saving ' + item + '.pkl')
        new_df.to_pickle(item + '.pkl')






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--linux_dir', type=str, default='dataset/linux-5.9-rc1/', help='path to linux sources')
    args = parser.parse_args()
    path = args.linux_dir

    filetypes_to_use = ['c', 'h', 'sh']

    num_workers = 8

    run(path, filetypes_to_use, num_workers)
