import glob, json, os, argparse
from tqdm import tqdm

from transformers import GPT2Tokenizer
import pandas as pd

from torch.utils.data.dataset import Dataset

import sklearn.model_selection as model_selection

import random
import time
import torch


def load_pickles(languages):
    df = pd.DataFrame()
    print('loading pickles...')
    for lang in languages:

        path = 'dataset/' + lang + '.pkl'
        print('reading ' + path)
        new_df = pd.read_pickle(path)  # todo: coerce to utf-8? is that done automatically here?

        df = df.append(new_df)
    return df

def shuffle_dataset(df):
    print('shuffling dataset')
    # this means, return all rows, in random order. SHUFFLE
    # reset_index prevents pandas from creating a new column with the old index
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def split_data(df, validation_ratio=0.1):

    """
    does the train test validation split
    """
    train, test = model_selection.train_test_split(df, test_size=validation_ratio)

    print('training set size ' + str(train.shape))
    print('validation set size ' + str(test.shape))

    return train, test

def get_datasets(model):
    data_location = '/home/cameron/PycharmProjects/auto_coding/dataset/'
    languages = ['ruby', 'go']
    test_ratio = 0.1

    df = load_pickles(languages)
    df = shuffle_dataset(df)
    train_df, test_df = split_data(df, test_ratio)

    train_dataset = DatasetFromPandas(train_df, model)
    test_dataset = DatasetFromPandas(test_df, model)

    return train_dataset, test_dataset


class DatasetFromPandas(Dataset):
    def __init__(self, dataframe, model, segment_len=254, stride=10):
        """
        A dataset example where the class is embedded in the file names
        This data example also does not use any torch transforms
        """
        self.df = dataframe
        self.current_iteration = 0
        self.stride = stride
        self.segment_len = segment_len
        self.model = model

        self.inputs = []
        # load_cache = False
        # if cache_path != None:
        #     load_cache = self._load_cache(cache_path)
        # if not load_cache:
        #     self._build(file_path, model)
        # if cache_path != None:
        #     self._cache(cache_path)

        data_top = self.df.columns.tolist()
        print('dataset columns ' + str(data_top))

    def get_next(self):
        item = self.__getitem__(self.current_iteration)
        self.current_iteration += 1
        return item

    def __getitem__(self, index):
        """
        gets specified dataset item and then
        segments and picks a random segment
        this is done to save memory
        """

        item = self.df.iloc[index]
        code = item['code']
        lang = item['language']

        encoded = self.model.tokenizer.encode(code) #todo: we are tokenizing here when this could be done before, and in fact it might already be done

        if lang == "python":
            encoded_plus = self.model.tokenizer.encode_plus(
                self.model.tokenize("<python>") + encoded + [self.model.eos_token_id],
                max_length=self.model.max_seq_length)

        elif lang == "java":
            encoded_plus = self.model.tokenizer.encode_plus(
                self.model.tokenize("<java>") + encoded + [self.model.eos_token_id],
                max_length=self.model.max_seq_length)

        elif lang == "javascript":
            encoded_plus = self.model.tokenizer.encode_plus(
                self.model.tokenize("<javascript>") + encoded + [self.model.eos_token_id],
                max_length=self.model.max_seq_length)

        elif lang == "php":
            encoded_plus = self.model.tokenizer.encode_plus(
                self.model.tokenize("<php>") + encoded + [self.model.eos_token_id],
                max_length=self.model.max_seq_length)

        elif lang == "ruby":
            encoded_plus = self.model.tokenizer.encode_plus(
                self.model.tokenize("<ruby>") + encoded + [self.model.eos_token_id],
                max_length=self.model.max_seq_length)

        elif lang == "go":
            encoded_plus = self.model.tokenizer.encode_plus(
                self.model.tokenize("<go>") + encoded + [self.model.eos_token_id],
                max_length=self.model.max_seq_length)


        #break it into segments, then pick a random one
        segments = {}
        for i in range(len(encoded) // self.stride):
            seg = encoded[i * self.stride:i * self.stride + self.segment_len]
            segments.update({"input_ids": seg})

        item = random.choice(segments)
        # item = {'features': item, 'labels': None}  #todo: why do i have to do this, i think it has to do with caching


        return item

    def __len__(self):
        return self.df.shape[0]

if __name__ == '__main__':

    """
    tests the dataset thing
    """

    from model import GPTSingleHead


    MODEL_MAP = {"distilgpt2": "distilgpt2", "gpt2": "gpt2", "gpt2_medium": "gpt2-medium",
                 "gpt2_large": "gpt2-large"}

    dataset_folder = f"/home/cameron/PycharmProjects/auto_coding/dataset/source_code/json/"
    file_path = dataset_folder + "train.jsonl"
    output_path = f"./model/distilgpt2_fine_tuned_coder"
    data_location = '/home/cameron/PycharmProjects/auto_coding/dataset/'

    model = GPTSingleHead(MODEL_MAP['distilgpt2'], max_seq_length=256)


    #languages = ['python', 'javascript', 'java', 'php', 'ruby', 'go']
    languages = ['javascript', 'ruby']
    test_ratio = 0.1

    df = load_pickles(languages)
    df = shuffle_dataset(df)
    train_df, test_df = split_data(df, test_ratio)

    train_dataset = DatasetFromPandas(train_df, model)
    test_dataset = DatasetFromPandas(test_df, model)

    while True:
        ass = train_dataset.get_next()
        print(ass)
        time.sleep(1)


