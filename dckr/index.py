
"""

https://www.wintellect.com/containerize-python-app-5-minutes/

"""

from typing import List, Dict
import torch
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

from flask import Flask

from flask import request

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import argparse



# code for running the pytorch thing


logging.basicConfig(
    format=logging.BASIC_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Params')
parser.add_argument('--model_path', type=str, default="model/distilgpt2_fine_tuned_coder/0_GPTSingleHead/",
                    help='the path to load fine-tuned model')
parser.add_argument('--max_length', type=int, default=64,
                    help='maximum length for code generation')
parser.add_argument('--temperature', type=float, default=0.7,
                    help='temperature for sampling-based code geneeration')
parser.add_argument(
    "--use_cuda", action="store_true", help="inference with gpu?"
)

langs = ["<python>", "<javascript>", "<java>", "<php>", "<ruby>", "<go>", "<c>", "<h>", "<sh>"]

args = parser.parse_args()

# load fine-tunned model and tokenizer from path
model = GPT2LMHeadModel.from_pretrained(args.model_path)
tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)

model.eval()
if args.use_cuda:
    model.to("cuda")




# code for the flask interface

app = Flask(__name__)

"""
{"language":"Python","framework":"Flask","website":"Scotch","version_info":{"python":3.4,"flask":0.12},"examples":["query","form","json"],"boolean_test":true}





{
  "language": "python",
  "text": "Flask",
  "website": "Scotch",
  "version_info": {
    "python": 3.4,
    "flask": 0.12
  },
  "examples": [
    "query",
    "form",
    "json"
  ],
  "boolean_test": true
}

{
  "language": "python",
  "text": "def bubble"
  }
}

"""


@app.route('/json-example', methods=['POST']) #GET requests will be blocked
def json_example():
    req_data = request.get_json()

    language = req_data['language']
    text = req_data['text']

    print(str(req_data))

    print(language)
    print(text)

    input_ids = tokenizer.encode("<" + str(language).lower() + "> " + text, return_tensors='pt')

    outputs = model.generate(input_ids=input_ids.to("cuda") if args.use_cuda else input_ids,
                             max_length=args.max_length,
                             temperature=args.temperature,
                             num_return_sequences=1)
    print(type(outputs))

    for i in range(1):
        decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
        print(decoded)

    return decoded

@app.route("/")
def hello():

    lang = 'python'

    context = 'exit'

    input_ids = tokenizer.encode("<" + str(lang).lower() + "> " + context, return_tensors='pt')

    outputs = model.generate(input_ids=input_ids.to("cuda") if args.use_cuda else input_ids,
                             max_length=args.max_length,
                             temperature=args.temperature,
                             num_return_sequences=1)
    print(type(outputs))

    for i in range(1):
        decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
        print(decoded)

    return decoded

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug=True)