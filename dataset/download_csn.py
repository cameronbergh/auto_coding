#!/usr/bin/env python
"""

    # This code is copied and modified from the CodeSearchNet repository (https://github.com/github/CodeSearchNet)
    # CodeSearchNet is provided with the MIT License

    this program will download the CodeSearchNet dataset

Usage:
    download_csn.py --dir DESTINATION_DIR

    example:
        "python download_csn.py"
        "python download_csn.py --dir ./csn_data/"


Options:
    -h --help   Show this screen.
"""

import os
from subprocess import call
import argparse

# when training language models, there was some paper (i think from google?) that said:
# when training translation models, adding more languages to the model improved its performance
# on languages that it wasnt even being tested on? or something like that? basically, there is a synergy effect.
# so it would be neat to find out if that same thing happens with programming languages

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--dir', type=str, default='./csn_data/',
                        help='where to save the CodeSearchNet data')
    args = parser.parse_args()

    print(args)

    destination_dir = os.path.abspath(args.dir)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    os.chdir(destination_dir)

    for language in ('python', 'javascript', 'java', 'ruby', 'php', 'go'):
        call(['wget', 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{}.zip'.format(language), '-P', destination_dir, '-O', '{}.zip'.format(language)])
        call(['unzip', '{}.zip'.format(language)])
        call(['rm', '{}.zip'.format(language)])
