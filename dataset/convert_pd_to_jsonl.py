import glob, json, os, argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import GPT2Tokenizer
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--segment_len', type=int, default=254,
                        help='the length of each example')
    # we set this to be 254 instead of 256 because we want the input to be like: <control_code> input_ids <eos>
    parser.add_argument('--stride', type=int, default=10,
                        help='stride to split training examples')
    parser.add_argument('--dev_size', type=float, default=0.1,
                        help='split ratio of development set for each language')
    args = parser.parse_args()


    # languages = ['python', 'javascript', 'java', 'php', 'ruby', 'go']
    languages = ['php', None]


    for lang in languages:

        gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=False)

        segments = {}

        path = lang + '.pkl'
        print('Reading pkl ' + path)
        df = pd.read_pickle(path)  #todo: coerce to utf-8?

        for each_src in tqdm(df['code']):

            code_content = each_src
            encoded = gpt2_tok.encode(code_content) # the original code ignores the "Token indices sequence length..." error
            for i in range(len(encoded) // args.stride):
                seg = encoded[i * args.stride:i * args.stride + args.segment_len]
                if path not in segments:
                    segments[path] = []
                segments[path].append(json.dumps({"token_ids": seg, "label": lang}))

        train, dev = [], []
        for key in segments:
            # we don't shuffle before splitting because we want the train and dev to be very different (less overlapping)
            tr, de = train_test_split(segments[key], test_size=args.dev_size)
            train += tr
            dev += de

        to_path = os.path.join("dataset", str(lang))

        if not os.path.isdir(to_path):
            os.makedirs(to_path)

            #todo: this is going to use up TERABYTES of disk

        print('writing train.jsonl')
        with open(os.path.join(to_path, "train.jsonl"), "w") as f:
            #without this for loop, we get a MemoryError
            for item in train:
                f.write("\n" + item)

        print('writing dev.jsonl')
        with open(os.path.join(to_path, "dev.jsonl"), "w") as f:
            for item in dev:
                f.write("\n" + item)

        del gpt2_tok

