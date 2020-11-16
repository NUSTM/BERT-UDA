import numpy as np
import json
import os
from collections import defaultdict, namedtuple
import random
import spacy
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
import torch


class ABSATokenizer(BertTokenizer):
    '''
    The text should have been pre-processed before, only do sub word tokenizer here.
    '''
    def subword_tokenize(self, tokens, labels, tag):  # for AE
        split_tokens, split_labels, split_tags = [], [], []
        idx_map = []
        for ix, token in enumerate(tokens):
            sub_tokens = self.wordpiece_tokenizer.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)

                if labels[ix].startswith('B') and jx != 0:
                    split_labels.append(labels[ix].replace('B', 'I'))
                else:
                    split_labels.append(labels[ix])

                split_tags.append(tag[ix])
                idx_map.append(ix)
        return split_tokens, split_labels, split_tags, idx_map

    def convert_tag_to_ids(self, tags, tag_vocab_dict):
        # input: tag list return: index list
        new_tags = []
        for t in tags:
            if t in tag_vocab_dict.keys():
                new_tags.append(t)
            else:
                new_tags.append('[UNK]')
        return [tag_vocab_dict[t] for t in new_tags]

    def convert_dep_to_ids(self, rel, dep_vocab_dict):
        # input: tag list return: index list
        new_tags = []
        for t in rel:
            if t in dep_vocab_dict or t == '-1' or t == -1:
                new_tags.append(t)
            else:
                new_tags.append('[UNK]')
        return [dep_vocab_dict.get(t, -1) for t in new_tags]


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, tag=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label
        self.tag = tag


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, tag_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.tag_id = tag_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)
# 892:1***Boot time is super fast , around anywhere from 35 seconds to 1 minute .***B I O O O O O O O O O O O O O***NN NN VBZ JJ RB , IN RB IN CD NNS TO CD NN .

    def read_txt(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as fp:
            text = fp.readlines()
        lines = {}
        id = 0
        for _, t in enumerate(text):
            try:
                sentence, label, tag_label = t.split('***')
            except:
                sentence, label = t.split('***')
                tag_label = label
            label = label.split()
            sentence = sentence.lower().split()
            tag_label = tag_label.split()
            assert len(label) == len(sentence) == len(tag_label), print(sentence, label)
            lines[id] = {'sentence': sentence, 'label': label, 'tag': tag_label}
            id += 1
        return lines


class ABSAProcessor(DataProcessor):
    """Processor for the SemEval Aspect Extraction and end2end absa ."""

    def get_train_examples(self, data_dir, task_type, fn=".train.txt"):
        """See base class."""
        source, target = data_dir.split('-')
        return self._create_examples(
            self.read_txt(os.path.join('./datasets', source + fn)), task_type= task_type, set_type="train")

    def get_dev_examples(self, data_dir, task_type, fn=".train.txt"):
        """See base class."""
        source, target = data_dir.split('-')
        return self._create_examples(
        	self.read_txt(os.path.join('./datasets', source + fn)), task_type= task_type, set_type="dev")

    def get_test_examples(self, data_dir, task_type, fn=".test.txt"):
        """See base class."""
        source, target = data_dir.split('-')
        return self._create_examples(
            self.read_txt(os.path.join('./datasets', target + fn)), task_type= task_type, set_type="test")

    def get_labels(self, task_type='ae'):
        """See base class."""
        task_type = task_type.lower()
        assert task_type in {'ae', 'absa'}, print('unknow task type ! please choose in [ae, absa]')
        labels = ['O', 'B', 'I'] if task_type == 'ae' else ['O', 'B-POS', 'B-NEG', 'B-NEU', 'I-POS', 'I-NEG', 'I-NEU']
        assert labels[0] == 'O', print('O not in labels! please make sure O in label !')
        return labels

    def ot2bio(self, ts_tag_sequence):
        """
        ot2bio function for ts tag sequence
        :param ts_tag_sequence:
        :return: BIO labels for aspect extraction
        """
        new_ts_sequence = []
        n_tag = len(ts_tag_sequence)
        prev_pos = 'O'
        for i in range(n_tag):
            cur_ts_tag = ts_tag_sequence[i]
            if 'T' not in cur_ts_tag:
                new_ts_sequence.append('O')
                cur_pos = 'O'
            else:
                cur_pos, cur_sentiment = cur_ts_tag.split('-')
                if prev_pos != 'O':  # cur_pos == prev_pos
                    # prev_pos is T
                    new_ts_sequence.append('I') 
                else:
                    new_ts_sequence.append('B')
            prev_pos = cur_pos
        return new_ts_sequence

    def ot2bio_absa(self, ts_tag_sequence):
        """
        ot2bio function for ts tag sequence
        :param ts_tag_sequence:
        :return: BIO-{POS, NEU, NEG} for end2end absa.
        """
        new_ts_sequence = []
        n_tag = len(ts_tag_sequence)
        prev_pos = 'O'
        for i in range(n_tag):
            cur_ts_tag = ts_tag_sequence[i]
            if 'T' not in cur_ts_tag:
                new_ts_sequence.append('O')
                cur_pos = 'O'
            else:
                cur_pos, cur_sentiment = cur_ts_tag.split('-')
                if prev_pos != 'O':  # cur_pos == prev_pos
                    # prev_pos is T
                    new_ts_sequence.append('I-%s' % cur_sentiment)  # I 'I-%s' % cur_sentiment
                else:
                    new_ts_sequence.append('B-%s' % cur_sentiment)
            prev_pos = cur_pos
        return new_ts_sequence

    def _create_examples(self, lines, task_type='ae', set_type=''):
        """Creates examples for the training and dev sets."""
        task_type = task_type.lower()
        assert task_type in {'ae', 'absa'}, print('unknow task type ! please choose in [ae, absa]')
        examples = []
        ids = 0
        for i in range(len(lines)):
            guid = "%s-%s" % (set_type, ids)
            text_a = lines[i]['sentence']
            label = self.ot2bio(lines[i]['label']) if task_type == 'ae' else self.ot2bio_absa(lines[i]['label'])
            tag = lines[i]['tag']
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label, tag=tag))
            ids += 1
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""  
    PAD_TOKEN_LABEL = -1 
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    label_map['subwords'] = PAD_TOKEN_LABEL

    with open('./datasets/tag_vocab.txt', 'r', encoding='utf-8') as fp:
            tag_vocab = fp.read().splitlines()
    tag_vocab_dict = dict(zip(tag_vocab, [i for i in range(len(tag_vocab))]))

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a, labels_a, tag_a, example.idx_map = tokenizer.subword_tokenize(
            [token.lower() for token in example.text_a], example.label, example.tag)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
            tag_a = tag_a[0:(max_seq_length - 2)]
            labels_a = labels_a[0:(max_seq_length - 2)]
        assert len(tokens_a) == len(tag_a) == len(labels_a)

        tokens = []
        tags = []
        segment_ids = []
        tokens.append("[CLS]")
        tags.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        for tag in tag_a:
            tags.append(tag)
        tokens.append("[SEP]")
        tags.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        tag_ids = tokenizer.convert_tag_to_ids(tags, tag_vocab_dict)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            tag_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(tag_ids) == max_seq_length

        label_id = [PAD_TOKEN_LABEL] * len(input_ids)  # -1 is the index to ignore use 0
        # truncate the label length if it exceeds the limit.
        lb = [label_map[label] for label in labels_a]
        if len(lb) > max_seq_length - 2:
            lb = lb[0:(max_seq_length - 2)]
        label_id[1:len(lb) + 1] = lb  # 前后都是-1

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                tag_id=tag_ids))
    return features

AuxInputFeatures = namedtuple("AuxInputFeatures", "input_ids input_tags head_tokens head_rel_labels input_mask lm_label_ids tag_label_ids domain_label")


def aux_convert_example_to_features(example, tokenizer, max_seq_length, tag_vocab, dep_vocab):
    tokens = example["tokens"]
    masked_lm_positions = example.get("masked_lm_positions", [])
    masked_lm_labels = example.get("masked_lm_labels", [])
    tags = example["tags"]
    masked_tag_positions = example.get("masked_tag_positions", [])
    masked_tag_labels = example.get("masked_tag_labels", [])
    head_tokens = example["head_tokens"]
    head_rel = example["arc_rel"]
    domain_label = 1 if example["domain_label"] == 'source' else 0 
    assert len(tokens) == len(tags) == len(head_tokens) == len(head_rel) <= max_seq_length, print(len(tokens), len(tags), len(head_tokens))  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
    input_tags = tokenizer.convert_tag_to_ids(tags, tag_vocab)
    masked_tag_labels = tokenizer.convert_tag_to_ids(masked_tag_labels, tag_vocab)
    head_rel_labels = tokenizer.convert_dep_to_ids(head_rel, dep_vocab)
    head_tokens = [int(index) for index in head_tokens]  # 将字符串转换成整数，存放的是head token index, ~(0, max_seq_len)

    input_array = np.zeros(max_seq_length, dtype=np.int) 
    input_array[:len(input_ids)] = input_ids

    tag_array = np.zeros(max_seq_length, dtype=np.int)  
    tag_array[:len(input_tags)] = input_tags

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_tags)] = 1

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    tag_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    tag_label_array[masked_tag_positions] = masked_tag_labels

    head_tokens_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    head_tokens_array[:len(head_tokens)] = head_tokens

    head_rel_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    head_rel_array[:len(head_rel_labels)] = head_rel_labels
 
    features = AuxInputFeatures(input_ids=input_array,
                             input_tags=tag_array,
                             head_tokens=head_tokens_array,
                             head_rel_labels=head_rel_array,
                             input_mask=mask_array,
                             lm_label_ids=lm_label_array,
                             tag_label_ids=tag_label_array,
                             domain_label=domain_label)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs=1):
        # self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = os.path.join(training_path, f"epoch_{self.data_epoch}.json")
        metrics_file = os.path.join(training_path, f"epoch_{self.data_epoch}_metrics.json")
        assert os.path.isfile(data_file) and os.path.isfile(metrics_file), print(data_file, metrics_file)
        metrics = json.loads(open(metrics_file, 'r').read())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        with open('./datasets/tag_vocab.txt', 'r', encoding='utf-8') as fp:
            self.tag_vocab = fp.read().splitlines()
            self.tag_vocab_dict = dict(zip(self.tag_vocab, [i for i in range(len(self.tag_vocab))]))
        with open('./datasets/dep_vocab.txt', 'r', encoding='utf-8') as fp:
            self.dep_vocab = fp.read().splitlines()
            self.dep_vocab_dict = dict(zip(self.dep_vocab, [i for i in range(len(self.dep_vocab))]))

        input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
        tag_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)  
        input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
        lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)  
        tag_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)  
        head_tokens_index = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)  
        rel_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)  
        domain_labels = np.zeros(shape=(num_samples,), dtype=np.bool)
        # logging.info(f"Loading training examples for epoch {epoch}")
        with open(data_file, 'r') as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = aux_convert_example_to_features(example, tokenizer, seq_len, self.tag_vocab_dict, self.dep_vocab_dict)
                input_ids[i] = features.input_ids
                tag_ids[i] = features.input_tags
                head_tokens_index[i] = features.head_tokens
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                tag_label_ids[i] = features.tag_label_ids
                rel_label_ids[i] = features.head_rel_labels
                domain_labels[i] = features.domain_label
        assert i == num_samples - 1  # Assert that the sample count metric was true
        # logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.tag_ids = tag_ids
        self.head_tokens_index = head_tokens_index
        self.rel_label_ids = rel_label_ids
        self.input_masks = input_masks
        self.lm_label_ids = lm_label_ids
        self.tag_label_ids = tag_label_ids
        self.domain_labels = domain_labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.tag_ids[item].astype(np.int64)),
                torch.tensor(self.head_tokens_index[item].astype(np.int64)),
                torch.tensor(self.rel_label_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.tag_label_ids[item].astype(np.int64)),
                torch.tensor(self.domain_labels[item].astype(np.int64)))
