from torch.utils.data import Dataset, IterableDataset, DataLoader
import os, pickle, json
import logging

import pandas as pd

import sklearn.model_selection as model_selection

logger = logging.getLogger(__name__)

from dpu_utils.utils import RichPath

from tqdm import tqdm

import random

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

        #tokenize the code
        encoded = self.model.tokenizer.encode(code)

        #make it into a bunch of properly sized sequences
        segments = []
        for i in range(len(encoded) // self.stride):
            seg = encoded[i * self.stride:i * self.stride + self.segment_len]
            segments.append({"token_ids": seg, "label": lang})

        #if there is more than one sequence made, then pick a random one
        # (this is done to save memory) #todo: prove doing it this way is okay
        if len(encoded) // self.stride == 0:
            item = [encoded]
        else:
            item = random.choice(segments)

        #add special tokens
        encoded_plus = self.model.tokenizer.encode_plus(
            self.model.tokenize("<" + lang + ">") + item['token_ids'] + [self.model.eos_token_id])

        # remove the attention mask data from the dict
        del encoded_plus.data['attention_mask']

        return encoded_plus.data

    def __len__(self):
        return self.df.shape[0]

class SrcCodeDataset(Dataset):
    def __init__(self, file_path, model, cache_path=None):
        """
        this dataset class is used to load source code dataset in batch for fine-tuning with GPT2LMModel
        :param model: the model that the dataset will be fed to
        """
        self.inputs = []
        load_cache = False
        if cache_path != None:
            load_cache = self._load_cache(cache_path)
        if not load_cache:
            self._build(file_path, model)
        if cache_path != None:
            self._cache(cache_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ids = self.inputs[index]["input_ids"]
        # input_mask = self.inputs[index]["attention_mask"] we don't need attention_mask for this task
        # return {"input_ids": input_ids, "input_mask": input_mask}
        return {"input_ids": input_ids}

    def _load_cache(self, cache_path):
        load_cache = False
        if os.path.isdir(cache_path):
            if os.path.isfile(os.path.join(cache_path, "inputs.pk")):
                with open(os.path.join(cache_path, "inputs.pk"), "rb") as f:
                    logger.info(
                        f"  load cached token ids of model from {cache_path}")
                    self.inputs = pickle.load(f)
                    load_cache = True
        return load_cache

    def _cache(self, cache_path):
        if not os.path.isdir(cache_path):
            os.makedirs(cache_path)
        with open(os.path.join(cache_path, "inputs.pk"), "wb") as f:
            pickle.dump(self.inputs, f)
            logger.info(
                f"  save tokenized ids of samples to: {cache_path}/inputs.pk")

    def _build(self, file_path, model):
        with open(file_path) as f:

            print(file_path)
            for line in tqdm(f):
                example = json.loads(line.strip())

                if example["label"].lower() == "python":
                    encoded_plus = model.tokenizer.encode_plus(
                        model.tokenize("<python>") + example["token_ids"] + [model.eos_token_id],
                        max_length=model.max_seq_length)
                elif example["label"].lower() == "java":
                    encoded_plus = model.tokenizer.encode_plus(
                        model.tokenize("<java>") + example["token_ids"] + [model.eos_token_id],
                        max_length=model.max_seq_length)

                self.inputs.append(encoded_plus.data)

if __name__ == '__main__':

    """ 
    this is for unit testing? 
    """

    from model import GPTSingleHead

    MODEL_MAP = {"distilgpt2": "distilgpt2", "gpt2": "gpt2", "gpt2_medium": "gpt2-medium",
                 "gpt2_large": "gpt2-large"}

    model = GPTSingleHead(MODEL_MAP['distilgpt2'], max_seq_length=256)
    model.add_special_words({"pad_token": "<pad>", "additional_special_tokens": ["<ruby>", "<javascript>"]})


    #languages = ['javascript', 'ruby']
    languages = ['python', 'javascript', 'java', 'php', 'ruby', 'go']


    test_ratio = 0.1


    df = load_pickles(languages)
    df = shuffle_dataset(df)
    train_df, test_df = split_data(df, test_ratio)

    train_dataset = DatasetFromPandas(train_df, model)
    test_dataset = DatasetFromPandas(test_df, model)

    train_dataset = DatasetFromPandas(train_df, model)


    for i in range(0, 100):
        item = train_dataset.__getitem__(i)
        #print(item)
        print(model.tokenizer.decode(item['input_ids'], skip_special_tokens=False))
        print(len(item['input_ids']))
        print(i)




