import os
import sys
sys.path.append('../')
import logging
import argparse
import random
import json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

from pytorch_transformers.tokenization_bert import BertTokenizer
from optimization import BertAdam

import data_utils
from data_utils import ABSATokenizer, aux_convert_example_to_features, PregeneratedDataset
import modelconfig
from model import *
from eval import eval_result, eval_ts_result
from collections import namedtuple
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)


def train(args):
    processor = data_utils.ABSAProcessor()
    label_list = processor.get_labels('absa')
    model = ABSABert.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model],
                                                    num_labels=len(label_list))
    tokenizer = ABSATokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    model.cuda()

    epoch_dataset = PregeneratedDataset(epoch=0, training_path=args.data_dir, tokenizer=tokenizer,
                                                num_data_epochs=1)
    unlabel_train_sampler = RandomSampler(epoch_dataset)
    unlabel_train_dataloader = DataLoader(epoch_dataset, sampler=unlabel_train_sampler, batch_size=args.train_batch_size)
    unlabel_iter = iter(unlabel_train_dataloader)
    train_steps = len(unlabel_train_dataloader)

    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad == True]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = train_steps * args.num_train_epochs
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total
                         )

    global_step = 0
    model.train()
    total_loss = 0
    for e_ in range(args.num_train_epochs):
        if e_ > 0:
            epoch_dataset = PregeneratedDataset(epoch=e_, training_path=args.data_dir, tokenizer=tokenizer,
                                                num_data_epochs=1)
            unlabel_train_sampler = RandomSampler(epoch_dataset)
            unlabel_train_dataloader = DataLoader(epoch_dataset, sampler=unlabel_train_sampler, batch_size=args.train_batch_size)
            unlabel_iter = iter(unlabel_train_dataloader)
            train_steps = len(unlabel_train_dataloader)
            logger.info('unlabel data number:{} steps：{}'.format(len(epoch_dataset), train_steps))

        for step in range(train_steps):       
            batch = unlabel_iter.next()
            batch = tuple(t.cuda() for t in batch)
            input_ids, tag_ids, head_tokens_index, rel_label_ids, input_mask, lm_label_ids, tag_label_ids, _ = batch
            loss = model(input_ids, input_tags=tag_ids, head_tokens_index=head_tokens_index, dep_relation_label=rel_label_ids, 
                    masked_tag_labels=tag_label_ids, attention_mask=input_mask)

            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
                
            global_step += 1
            if global_step % 100 == 0:
                logger.info('in step {} loss is: {}'.format(global_step, total_loss / global_step))
            # >>>> perform validation at the end of each epoch .
        os.makedirs(args.output_dir + '/epoch' + str(e_), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir + '/epoch' + str(e_), "model.pt"))


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default='bert_base', type=str)

    parser.add_argument("--data_dir",
                        default='./datasets/unlabel/rest-laptop-aux',
                        type=str,
                        required=False,
                        help="The input data dir containing json files.")

    parser.add_argument("--output_dir",
                        default='./out_feature_models/base-rest-laptop',
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument('--seed',
                        type=int,
                        default=22,
                        help="random seed for initialization")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=100,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_valid",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--train_batch_size",  
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,  
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_epochs",
                        default=5,  
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        train(args)

if __name__ == "__main__":
    main()

