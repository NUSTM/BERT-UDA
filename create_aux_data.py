from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from tempfile import TemporaryDirectory
import shelve
from multiprocessing import Pool

import random
from random import randrange, randint, shuffle, choice
from pytorch_transformers.tokenization_bert import BertTokenizer
import numpy as np
import json
import collections
import os
import spacy
import glob


class SubTokenizer(BertTokenizer):
    '''
    bert toknizer： only do sub word tokenizer here
    '''
    def subword_tokenize(self, tokens, labels, head, arc_label):  # for tag split
        # input : tokens list, tag list head list; output: token list, tag list, head_index, idx_map;
        split_tokens, split_labels = [], []
        idx_map = []
        head_index, split_arc_label = [], []
        first_token_map = dict()
        for ix, token in enumerate(tokens):
            sub_tokens = self.wordpiece_tokenizer.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)

                if jx == 0:
                    split_labels.append(labels[ix] + '_B')  # the subwords share same pos tags, use noun_B, nonu_M, nonu_E to split 
                else:
                    split_labels.append(labels[ix] + '_I')

                # dependency relation and head index
                if jx == 0:
                    first_token_map[ix] = len(split_tokens) - 1  # a dict map original index to new index
                    split_arc_label.append(arc_label[ix])  
                    head_index.append(head[ix])  
                else:
                    head_index.append(-1)
                    split_arc_label.append(-1) 
                idx_map.append(ix)
        head_index = [first_token_map[i] if i != -1 else -1 for i in head_index]
        return split_tokens, split_labels, head_index, split_arc_label, idx_map


class DocumentDatabase:
    def __init__(self):
        # do not support reduce memory function
        self.documents = []
        self.documents_tags = [] 
        self.documents_head_index = []  
        self.documents_dep_rel = []  
        self.documents_domain_label = []
        self.document_shelf = None
        self.document_shelf_filepath = None
        self.temp_dir = None

        self.doc_lengths = []
        self.doc_cumsum = None
        self.cumsum_max = None

    def add_document(self, document, tags, head, rel, domain_label):
        if not document:
            return
        assert len(document) == len(tags) == len(head)
        self.documents.append(document)
        self.documents_tags.append(tags)
        self.documents_head_index.append(head)
        self.documents_dep_rel.append(rel)
        self.documents_domain_label.append(domain_label)
        self.doc_lengths.append(len(document))

    def _precalculate_doc_weights(self):
        self.doc_cumsum = np.cumsum(self.doc_lengths)
        self.cumsum_max = self.doc_cumsum[-1]

    def sample_doc(self, current_idx, sentence_weighted=True):
        if sentence_weighted:
            if self.doc_cumsum is None or len(self.doc_cumsum) != len(self.doc_lengths):
                self._precalculate_doc_weights()
            rand_start = self.doc_cumsum[current_idx]
            rand_end = rand_start + self.cumsum_max - self.doc_lengths[current_idx]
            sentence_index = randrange(rand_start, rand_end) % self.cumsum_max
            sampled_doc_index = np.searchsorted(self.doc_cumsum, sentence_index, side='right')
        else:
            # If we don't use sentence weighting, then every doc has an equal chance to be chosen
            sampled_doc_index = (current_idx + randrange(1, len(self.doc_lengths))) % len(self.doc_lengths)
        assert sampled_doc_index != current_idx
        return self.documents[sampled_doc_index], self.documents_tags[sampled_doc_index],
        self.documents_head_index[sampled_doc_index], self.documents_dep_rel[sampled_doc_index], \
        self.documents_domain_label[sampled_doc_index]

    def __len__(self):
        return len(self.doc_lengths)

    def __getitem__(self, item):
        return self.documents[item], self.documents_tags[item], self.documents_head_index[item], \
               self.documents_dep_rel[item], self.documents_domain_label[
                   item] 

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def truncate_seq_pair(tokens_a, tokens_b, tags_a, tags_b, head_a, head_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        if len(tokens_a) > len(tokens_b):
            trunc_tokens, trunc_tags, trunc_head = tokens_a, tags_a, head_a
        else:
            trunc_tokens, trunc_tags, trunc_head = tokens_b, tags_b, head_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
            del trunc_tags[0]
            del trunc_head[0]
        else:
            trunc_tokens.pop()
            trunc_tags.pop()
            trunc_head.pop()


def truncate_seq(tokens_a, tags_a, head_a, rel_a, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a)
        if total_length <= max_num_tokens:
            break

        trunc_tokens, trunc_tags, trunc_head, trunc_rel = tokens_a, tags_a, head_a, rel_a

        assert len(trunc_tokens) >= 1

        # 在列表末尾删除单词
        trunc_tokens.pop()
        trunc_tags.pop()
        trunc_head.pop()
        trunc_rel.pop()


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, tags, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list,
                                 tag_vocab):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []

    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.

        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])  # 保存了token index

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))

    shuffle(cand_indices)

    masked_lms = []
    masked_tags = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            masked_tag = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "[MASK]"
                masked_tag = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                    masked_tag = tags[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
                    masked_tag = choice(tag_vocab)

            masked_tags.append(MaskedLmInstance(index=index, label=tags[index]))
            tags[index] = masked_tag
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    masked_tags = sorted(masked_tags, key=lambda x: x.index)
    masked_tag_indices = [p.index for p in masked_tags]
    masked_tag_labels = [p.label for p in masked_tags]

    return tokens, mask_indices, masked_token_labels, tags, masked_tag_indices, masked_tag_labels


def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list, tag_vocab):
    """
    给定文档号，使用create_masked_lm_predictions构造Mask pos predict训练数据。
    """
    document, document_tag, document_head_index, documents_dep_rel, domain_label = doc_database[
        doc_idx]  # 返回文档内容和文档的词法标记

    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens

    instances = []
    current_chunk = []
    current_chunk_tags = []
    current_chunk_head_index = []
    current_chunk_rel = []
    current_length = 0
    ix = 0  
    while ix < len(document):  
        segment, segment_tag, segment_head_index, segment_rel = document[ix], document_tag[ix], document_head_index[ix], \
                                                                documents_dep_rel[ix]
        current_chunk.append(segment)
        current_chunk_tags.append(segment_tag)
        current_chunk_head_index.append(segment_head_index)
        current_chunk_rel.append(segment_rel)
        current_length += len(segment)
        if True: 
            if current_chunk:
                a_end = 1  
                tokens_a = []
                tags_a = []
                head_a = []
                rel_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                    tags_a.extend(current_chunk_tags[j])
                    head_a.extend(current_chunk_head_index[j])
                    rel_a.extend(current_chunk_rel[j])

                truncate_seq(tokens_a, tags_a, head_a, rel_a, max_num_tokens) 

                assert len(tokens_a) >= 1

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
                tags = ["[CLS]"] + tags_a + ["[SEP]"]  
                heads = [0] + [index + 1 if index < len(tokens_a) and index >= 0 else 0 for index in head_a] + [0]
                rel = [-1] + rel_a + [-1]  
                assert len(tokens) == len(heads) == len(tags)

                for i in range(len(heads)):
                    head_tokens = heads[i]
                    if head_tokens == 0:
                        rel[i] = -1  
                    assert head_tokens >= 0 and head_tokens < max_seq_length


                # 对语法序列进行破坏
                tokens, masked_lm_positions, masked_lm_labels, tags, masked_tag_positions, masked_tag_labels = create_masked_lm_predictions(
                    tokens, tags, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list, tag_vocab)

                assert len(heads) == len(tokens) == len(tags)

                instance = {
                    "tokens": tokens,
                    "head_tokens": heads,
                    "tags": tags,
                    "arc_rel": rel,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels,
                    "masked_tag_positions": masked_tag_positions,
                    "masked_tag_labels": masked_tag_labels,
                    "domain_label": domain_label[ix]
                }
                instances.append(instance)

            current_chunk = []
            current_chunk_tags = []
            current_chunk_head_index = []
            current_chunk_rel = []
            current_length = 0
        ix += 1

    return instances


def create_training_file(docs, vocab_list, args, epoch_num):
    epoch_filename = os.path.join(args.output_dir, "epoch_{}.json".format(epoch_num))
    num_instances = 0
    with open(epoch_filename, 'w') as epoch_file:
        for doc_idx in trange(len(docs), desc="Document"):
            doc_instances = create_instances_from_document(
                docs, doc_idx, max_seq_length=args.max_seq_len, short_seq_prob=args.short_seq_prob,
                masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list, tag_vocab=args.tag_vocab)
            doc_instances = [json.dumps(instance, ensure_ascii=False) for instance in doc_instances]
            for instance in doc_instances:
                epoch_file.write(instance + '\n')
                num_instances += 1
    metrics_file = os.path.join(args.output_dir, "epoch_{}_metrics.json".format(epoch_num))
    with open(metrics_file, 'w') as metrics_file:
        metrics = {"num_training_examples": num_instances, "max_seq_len": args.max_seq_len}
        metrics_file.write(json.dumps(metrics, ensure_ascii=False))


def parse_tree(doc):
    # get head index for each token 
    head_index = []
    for ix, t in enumerate(doc):
        prev = len(head_index)
        for jx, h in enumerate(doc):
            if t.head.text == h.text and t in [c for c in h.children]:
                head_index.append(jx)
                break
        if prev == len(head_index):  
            head_index.append(ix)
    assert len(head_index) == len(doc), print(len(head_index), len(doc))
    return head_index

def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=str, default='./datasets/unlabel/device-service-train-merge.txt', required=False, help="sentence in each line.")
    parser.add_argument("--output_dir", type=str, default='./datasets/unlabel/device-service-aux', required=False)
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased', required=False,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual-uncased", "bert-base-chinese", "bert-base-multilingual-cased"])
    parser.add_argument("--do_lower_case", default=True)  
    parser.add_argument("--do_whole_word_mask", default=True, 
                        help="Whether to use whole word masking rather than per-WordPiece masking.")

    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of workers to use to write the files")
    parser.add_argument("--epochs_to_generate", type=int, default=5, 
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=100) 
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=0.20,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=25,
                        help="Maximum number of tokens to mask in each sequence")

    args = parser.parse_args()
 
    with open('./datasets/tag_vocab.txt', 'r', encoding='utf-8') as fp:
        tag_vocab = fp.read().splitlines()
        tag_vocab = [l.strip() for l in tag_vocab]
    args.tag_vocab = tag_vocab

    tokenizer = SubTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())
    nlp = spacy.load("en_core_web_sm")
    all_tags = set()
    all_rel = set()
    with DocumentDatabase() as docs:
        for file in glob.glob(args.train_corpus):
            with open(file, 'r', encoding='utf-8') as f:
                doc = []  # token
                tag = []  # pos tags
                head = []  # head index
                arc_label = []  # dependency relation
                domain_label = []  # domain label
                for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
                    line = line.strip().lower() if args.do_lower_case else line.strip()
                    if len(line) == 1:
                        if len(doc): 
                            docs.add_document(doc, tag, head, arc_label, domain_label)
                        doc = []
                        tag = []
                        head = []
                        arc_label = []
                        domain_label = []
                    else:
                        domain, line = line.split('***')[:2]
                        nlp_doc = nlp(line)
                        tokens = [t.text for t in nlp_doc]
                        token_tags = [t.tag_ for t in nlp_doc]  
                        token_head = parse_tree(nlp_doc)  
                        token_dep_rel = [t.dep_ for t in nlp_doc]
                        all_rel.update(token_dep_rel)
                        tokens, token_tags, token_head_index, token_dep_rel, _ = tokenizer.subword_tokenize(tokens,
                                                                                                            token_tags,
                                                                                                            token_head,
                                                                                                            token_dep_rel)  # 只进行sub word tokenizer
                        all_tags.update(token_tags)
                        
                        assert len(tokens) == len(token_tags) == len(token_head_index) == len(token_dep_rel)
                        doc.append(tokens)
                        tag.append(token_tags)
                        head.append(token_head_index)
                        arc_label.append(token_dep_rel)
                        domain_label.append(domain)
                if doc:
                    docs.add_document(doc, tag, head, arc_label,
                                      domain_label)  # If the last doc didn't end on a newline, make sure it still gets added
        if len(docs) < 1:
            exit("ERROR: No document breaks were found in the input file! These are necessary to allow the script to "
                 "ensure that random NextSentences are not sampled from the same document. Please add blank lines to "
                 "indicate breaks between documents in your input file. If your dataset does not contain multiple "
                 "documents, blank lines can be inserted at any natural boundary, such as the ends of chapters, "
                 "sections or paragraphs.")
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

        if args.num_workers > 1:
            writer_workers = Pool(min(args.num_workers, args.epochs_to_generate))
            arguments = [(docs, vocab_list, args, idx) for idx in range(args.epochs_to_generate)]
            writer_workers.starmap(create_training_file, arguments)
        else:
            for epoch in trange(args.epochs_to_generate, desc="Epoch"):
                create_training_file(docs, vocab_list, args, epoch)


if __name__ == '__main__':
	random.seed(42)
	np.random.seed(42)
	main()
