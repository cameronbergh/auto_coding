
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


"""

notes from cameronbergh:

    - instead of using the train/test terminology, this project describes the datasets as train/dev.

    - I have added a dataset object which adds data from the CodeSearchNet dataset and also C code from the linux kernel
     im naming it the "csnl_dataset" in this program.

     -todo: the learning rate scheduler is too aggressibe for the csnl_dataset, find a better way.

     -to invoke the cnsl_dataset i have added some cmd arguments:
        --use_csnl_data and --csnl_data_dev_ratio 0.005

    -the csnl_dataset is shuffled at runtime, every time, which is different from the original code. it is also
     shuffled before the train/test split

    -the ".h" files from the linux kernel are actually C code but i thought it may be better to have the model
     learn them as a separa
    - I have added a dataset object which adds data from the CodeSearchNet dataset and also C code from the linux kernel
     im naming it the "csnl_dataset" in this program.

     -todo: the learning rate scheduler is too aggressibe for the csnl_dataset, find a better way.

     -to invoke the cnsl_dataset i have added some cmd arguments:
        --use_csnl_data and --csnl_data_dev_ratio 0.005

    -the csnl_dataset is shuffled at runtime, every time, which is diffete category.
     todo: experiment to find out if it performs better with mixed c/h files or with separated c/h files.

    -for the smaller model im going to exclude 's' files since those are assembly language and
    they look incomprehensible to me. though i can imagine some scenarios where they might be useful.

    -also, the <php> tag looks like something that might appear in places we didn't want it to.  <?php> is a legitimate
     tag in php, so i wonder if <php> appears  anywhere in the dataset.
     #todo: scan the dataset for "<php>" or any special tokens?

    - I didnt want to break the existing code so I made a argument --use_csnl_data which
     means the model will be trained on codesearchnet and also data from the linux kernel
     there was some broken code to do fuzzy matching and removing of duplicates from the dataset, but i removed it
     because it was broken and the duplication was minimal. todo: implement fuzzy matching dupe remover

    - the original code would take each code file and split it into some overlapping sequences with a stride of 10
     (not sure how to describe that) but that method would require terabytes of disk or memory. so i changed it to
     generate the sequences on-the-fly and pick a random one. ive seen this technique used in other projects but in
     this use-case i can imagine it might cause some problems.
        for example: the C and H files are often very large, while the codesearchnet files are much smaller,
        so maybe it will take many epochs for the model to learn the entire file? but maybe that's a good thing.

"""

import argparse, os
import logging

logging.basicConfig(
    format=logging.BASIC_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

MODEL_MAP = {"distilgpt2": "distilgpt2", "gpt2": "gpt2", "gpt2_medium": "gpt2-medium",
             "gpt2_large": "gpt2-large"}
from model import GPTSingleHead
from trainer import ModelTrainer
from data import CSNL_Dataset, SrcCodeDataset, load_pickles, split_data, shuffle_dataset

from evaluate import SingleCLMEvaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper params')
    parser.add_argument('--model_select', type=str, default="distilgpt2",
                        help='model select from distilgpt2, gpt2_medium, gpt2, or gpt2_large')
    parser.add_argument('--dataset_name', type=str, default="source_code",
                        help='dataset name whatever name you put into the ./dataset directory (by default: source_code)')
    parser.add_argument('--per_gpu_train_batch_size', type=int, default=6,
                        help='input batch size for training')
    parser.add_argument('--dev_batch_size', type=int, default=8,
                        help='input batch size for development')
    parser.add_argument('--num_epochs_train', type=int, default=16,
                        help='number of epochs to train')
    parser.add_argument('--max_seq_length', type=int, default=256,
                        help=' maximum sequence length of samples in a batch for training')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.2,
                        help='warmup_ratio')
    parser.add_argument('--early_stop', type=int, default=20,
                        help='early_stop')
    parser.add_argument('--scheduler', type=str, default="warmuplinear",
                        help='scheduler')
    parser.add_argument('--seed', type=int, default=122,
                        help='random seed')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='accumulation steps if you want large batch size but can not fit in the memory allowed')
    parser.add_argument('--n_gpu', type=int, default=2,
                        help='number of gpu for training')
    parser.add_argument('--visiable_device', type=str, default="0,1",
                        help='visiable gpus for training, should be consistent with n_gpu')
    parser.add_argument('--evaluation_steps', type=int, default=5000,
                        help='evaluation_steps')
    parser.add_argument('--wandb_project_name', type=str, default="code_generate",
                        help='project name for wandb')
    parser.add_argument(
        "--restore_training", action="store_true", help="restore training if a saved checkopint exists"
    )
    parser.add_argument(
        "--with_wandb", action="store_true", help="Train with wandb tracking."
    )
    parser.add_argument(
        "--less_verbose", action="store_true", help="set logger to report only Errors. (not warnings)"
    )
    parser.add_argument('--csnl_dev_ratio', type=float, default=0.001,
                        help='ratio of dev (validation) data for the splitter')
    parser.add_argument(
        "--use_csnl_data", action="store_true",
        help="use camerons extended code dataset which includes CodeSearchNet and data from the linux kernel"
    )

    args = parser.parse_args()
    logger.info(f"args: {args}")
    dataset_folder = f"dataset/{args.dataset_name}/json/"
    assert args.model_select in MODEL_MAP.keys(), (f"model has to be in {MODEL_MAP.keys()}")
    output_path = f"model/{args.model_select}_fine_tuned_coder"
    logger.info("{} for dataset in: {}".format(output_path, dataset_folder))
    logger.info(
        f"*****************model select: {args.model_select} for code generation using dataset: {args.dataset_name}******************")

    # add more params for wandb
    args.wandb_run_name = output_path

    # initialize model by model name (the same as used in transformers lib)
    model = GPTSingleHead(MODEL_MAP[args.model_select], max_seq_length=args.max_seq_length)

    # add special tokens for controlling code generation by different programming language
    model.add_special_words({"pad_token": "<pad>",
                             "additional_special_tokens": ["<python>", "<javascript>", "<java>", "<php>", "<ruby>",
                                                           "<go>", "<c>", "<h>", "<sh>"]})

    # this sets the logger to report only errors and not the unneeded warnings that we get when doing tokenizing
    if args.less_verbose:
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    if args.use_csnl_data: #invoke the cnsl_dataset
        print('using csnl dataset')

        # get the cmd arg for cnsl_dataset's train/dev split
        csnl_dev_ratio = args.csnl_dev_ratio

        #specify the languages to load and look for, note: we have left out .pl and .s files
        languages = ['python', 'javascript', 'java', 'php', 'ruby', 'go', 'c', 'h', 'sh']
        #languages = ['python', 'javascript', 'java', 'php', 'ruby', 'go']
        #languages = ['c', 'h', 'sh']

        #load the pickle files created by the convert_csnl script and then shuffle them
        df = load_pickles(languages)
        df = shuffle_dataset(df)

        # do the train/dev split
        train_df, dev_df = split_data(df, csnl_dev_ratio)
        train_dataset = CSNL_Dataset(train_df, model)
        dev_dataset = CSNL_Dataset(dev_df, model)

    else: #if we want to use use the original dataset by CongCongWang

        print('using jsonl dataset')

        # load training dataset
        file_path = dataset_folder + "train.jsonl"
        train_dataset = SrcCodeDataset(file_path, model, cache_path=os.path.join(".cache", output_path, "train"))

        # load development (dev) dataset
        file_path = dataset_folder + "dev.jsonl"
        dev_dataset = SrcCodeDataset(file_path, model, cache_path=os.path.join(".cache", output_path, "dev"))

    # initialize development evaluator
    dev_evaluator = SingleCLMEvaluator()

    # initialize model trainer
    model_trainer = ModelTrainer(model,
                                 train_dataset=train_dataset,
                                 dev_dataset=dev_dataset,
                                 dev_evaluator=dev_evaluator,
                                 scheduler=args.scheduler,
                                 epochs=args.num_epochs_train,
                                 per_gpu_train_batch_size=args.per_gpu_train_batch_size,
                                 output_path=output_path,
                                 optimizer_params={'lr': args.lr, 'eps': 1e-6, 'correct_bias': False},
                                 evaluation_steps=args.evaluation_steps,
                                 early_stop=args.early_stop,
                                 dev_batch_size=args.dev_batch_size,
                                 restore_training=args.restore_training,
                                 accumulation_steps=args.accumulation_steps,
                                 n_gpu=args.n_gpu,
                                 visiable_device=args.visiable_device,
                                 warmup_ratio=args.warmup_ratio,
                                 seed=args.seed,
                                 data_loader_shuffle=True,
                                 wandb_config=args if args.with_wandb else None)
    # start training
    model_trainer.train()
