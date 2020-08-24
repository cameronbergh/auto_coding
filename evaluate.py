from typing import List, Dict
import torch
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    format=logging.BASIC_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Params')
    parser.add_argument('--model_path', type=str, default="model/distilgpt2_079/0_GPTSingleHead/",
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

    def lang_select():
        lang = ""
        while lang not in ["python", "javascript", "java", "php", "ruby", "go", "c", "h", "sh"]:
        #while lang not in ["python", "javascript", "java", "php", "ruby", "go"]:
            print('Enter the programming language you prefer (python, javascript, java, php, ruby, go, c, h, sh)')
            #print('Enter the programming language you prefer (python, javascript, java, php, ruby, go)')
            lang = input(">>> ").lower()
        return lang

            # print(len(code_content))
            # # if len(ass['code']) > 1:

    lang = lang_select()

    context = ""
    while context != "exit":
        print(f'You are using {lang} now. Enter the context code (exit or change_lang)')
        context = input(">>> ")

        if context == "change_lang":
            lang = lang_select()

            print(f"You are using {lang} now. Enter the context code")
            context = input(">>> ")

        input_ids = tokenizer.encode("<" + str(lang).lower() + "> " + context, return_tensors='pt')

        outputs = model.generate(input_ids=input_ids.to("cuda") if args.use_cuda else input_ids,
                                 max_length=args.max_length,
                                 temperature=args.temperature,
                                 num_return_sequences=1)

        print(type(outputs))

        for i in range(1):
            decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
            print(type(decoded))
            # ends with occurence of double new lines (to meet the convention of code completion)
            if "\n\n" in decoded:
                decoded = decoded[:decoded.index("\n\n")]
                print('Generated {}: {}'.format(i, decoded))


