#!/usr/bin/env python

# This code is copied and modified from the CodeSearchNet repository (https://github.com/github/CodeSearchNet)
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
Usage:
    download_csn.py DESTINATION_DIR

    python download_csn.py ./csn_data/

Options:
    -h --help   Show this screen.
"""

import os
from subprocess import call

from docopt import docopt


if __name__ == '__main__':
    args = docopt(__doc__)

    destination_dir = os.path.abspath(args['DESTINATION_DIR'])
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    os.chdir(destination_dir)

    #('python', 'javascript', 'java', 'ruby', 'php', 'go'):
    for language in ('python', 'java', 'ruby', 'php', 'go'):
        call(['wget', 'https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/{}.zip'.format(language), '-P', destination_dir, '-O', '{}.zip'.format(language)])
        call(['unzip', '{}.zip'.format(language)])
        call(['rm', '{}.zip'.format(language)])
