import json
import logging
import os
import pickle
import random

import pandas as pd
import sklearn.model_selection as model_selection
from dpu_utils.utils import RichPath
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def jsonl_to_df(input_folder: RichPath) -> pd.DataFrame:
    "Concatenates all jsonl files from path and returns them as a single pandas.DataFrame ."

    assert input_folder.is_dir(), 'Argument supplied must be a directory'
    dfs = []
    files = list(input_folder.iterate_filtered_files_in_dir('*.jsonl.gz'))
    assert files, 'There were no jsonl.gz files in the specified directory.'
    print(f'reading files from {input_folder.path}')
    for f in tqdm(files, total=len(files)):
        dfs.append(
            pd.DataFrame(list(f.read_as_jsonl(error_handling=lambda m, e: print(f'Error while loading {m} : {e}')))))
    return pd.concat(dfs)


# loads picklled dataframes from disk, appends them all into one big dataframe
def load_pickles(languages):
    df = pd.DataFrame()
    print('loading pickles...')
    for lang in languages:
        path = 'dataset/' + lang + '.pkl'
        print('reading ' + path)
        new_df = pd.read_pickle(path)

        df = df.append(new_df)
    return df


def shuffle_dataset(df):
    """ takes all the code files in pandas and shuffles them """
    print('shuffling dataset')
    # this means, return all rows, in random order. SHUFFLE
    # reset_index prevents pandas from creating a new column with the old index
    df = df.sample(frac=1).reset_index(drop=True)
    return df


def split_data(df, validation_ratio=0.01):
    """ performs the train train/dev split """
    train, dev = model_selection.train_test_split(df, test_size=validation_ratio)

    print('training set size ' + str(train.shape))
    print('validation set size ' + str(dev.shape))

    return train, dev


class CSNL_Dataset(Dataset):
    def __init__(self, dataframe, model, segment_len=254, stride=10):
        """
        We call this the CSNL_Dataset,
        it is made of data from codesearchnet and some code from the linux kernel
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
        """

        while True:

            item = self.df.iloc[index]
            code_tokens = item['code']
            lang = item['language']

            # make the code sample into a bunch of properly sized sequences
            segments = []
            for i in range(len(code_tokens) // self.stride):
                seg = code_tokens[i * self.stride:i * self.stride + self.segment_len]

                # there was a rare case where the segment array would be empty so we do this check here
                if len(seg) > 0:
                    segments.append({"token_ids": seg, "label": lang})
                else:
                    continue

            # if there is more than one sequence made, then pick a random sample from the set
            if len(code_tokens) // self.stride == 0:
                item = code_tokens
            else:
                item = random.choice(segments)

            # add special tokens
            encoded_plus = self.model.tokenizer.encode_plus(
                self.model.tokenize("<" + str(lang) + ">") + item['token_ids'] + [self.model.eos_token_id])

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

def test_csnl_dataset_object():
    ###################################################
    # THIS CODE IS FOR TESTING THE CSNL_DATASET #######
    ################################################ df.to_pickle('linux.pkl')###

    from model import GPTSingleHead

    # set the logger to only report ERROR level messages, though, if there is any bizarre behavior this should be commented out
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    model = GPTSingleHead('distilgpt2', max_seq_length=256)
    model.add_special_words({"pad_token": "<pad>",
                             "additional_special_tokens": ["<python>", "<javascript>", "<java>", "<php>", "<ruby>",
                                                           "<go>", "<c>", "<h>", "<sh>"]})

    # languages = ['javascript', 'ruby']
    languages = ['h', 'c', 'sh' ]

    # dev_ratio = 0.001

    df = load_pickles(languages)
    # df = shuffle_dataset(df)
    # train_df, dev_df = split_data(df, dev_ratio)

    train_dataset = CSNL_Dataset(df, model)
    # dev_dataset = CSNL_Dataset(dev_df, model)

    batch_size = 1

    # dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)

    #print(len(train_dataset))

    i = 0
    for i in tqdm(range(0, (len(train_dataset) // batch_size))):
        item = next(iter(train_dataset))

        item = item['input_ids'] # get list of tensors

        print(model.tokenizer.decode(item, skip_special_tokens=False))
        print(i)


if __name__ == '__main__':
    test_csnl_dataset_object()

