# -*- coding: utf-8 -*-
import os
import numpy as np


def match(pred, gold):
    true_count = 0
    for t in pred:
        if t in gold:
            true_count += 1
    return true_count


def tag2aspect(tag_sequence):
    """
    convert BIO tag sequence to the aspect sequence
    :param tag_sequence: tag sequence in BIO tagging schema
    :return:
    """
    ts_sequence = []
    beg = -1
    for index, ts_tag in enumerate(tag_sequence):
        if ts_tag == 'O':
            if beg != -1:
                ts_sequence.append((beg, index-1))
                beg = -1
        else:
            cur = ts_tag.split('-')[0]  # unified tags
            if cur == 'B':
                if beg != -1:
                    ts_sequence.append((beg, index-1))
                beg = index

    if beg != -1:
        ts_sequence.append((beg, index))
    return ts_sequence


def tag2aspect_sentiment(ts_tag_sequence):
    '''
    support Tag sequence: ['O', 'B-POS', 'B-NEG', 'B-NEU', 'I-POS', 'I-NEG', 'I-NEU']
    '''
    ts_sequence, sentiments = [], []
    beg, end = -1, -1
    for index, ts_tag in enumerate(ts_tag_sequence):
        if ts_tag == 'O':
            if beg != -1:
                ts_sequence.append((beg, index-1, sentiments[0]))
                beg, sentiments = -1, []
        else:
            cur, pos = ts_tag.split('-')
            if cur == 'B':
                if beg != -1:
                    ts_sequence.append((beg, index-1, sentiments[0]))
                beg, sentiments = index, [pos]
            else:
                if beg != -1:
                    sentiments.append(pos)
    if beg != -1:
        ts_sequence.append((beg, index, sentiments[0]))
    return ts_sequence


def evaluate_chunk(test_Y, pred_Y):
    """
    evaluate function for aspect term extraction
    :param test_Y: gold standard tags (i.e., post-processed labels)
    :param pred_Y: predicted tags
    :return:
    """
    assert len(test_Y) == len(pred_Y)
    length = len(test_Y)
    TP, FN, FP = 0, 0, 0

    for i in range(length):
        gold = test_Y[i]
        pred = pred_Y[i]
        assert len(gold) == len(pred)
        gold_aspects = tag2aspect(gold)
        pred_aspects = tag2aspect(pred)
        n_hit = match(pred=pred_aspects, gold=gold_aspects)
        TP += n_hit
        FP += (len(pred_aspects) - n_hit)
        FN += (len(gold_aspects) - n_hit)
    precision = float(TP) / float(TP + FP + 0.00001)
    recall = float(TP) / float(TP + FN + 0.0001)
    F1 = 2 * precision * recall / (precision + recall + 0.00001)
    return precision, recall, F1


def evaluate_absa_chunk(test_Y, pred_Y):
    """
    evaluate function for end2end aspect based sentiment analysis, with labels: {B,I}-{POS, NEG, NEU} and O
    :param test_Y: gold standard tags (i.e., post-processed labels)
    :param pred_Y: predicted tags
    :return:
    """
    assert len(test_Y) == len(pred_Y)
    length = len(test_Y)
    TP, FN, FP = 0, 0, 0

    TP_A, FN_A, FP_A = 0, 0, 0

    for i in range(length):
        gold = test_Y[i]
        pred = pred_Y[i]
        assert len(gold) == len(pred)
        gold_aspects = tag2aspect_sentiment(gold)
        pred_aspects = tag2aspect_sentiment(pred)
        n_hit_a = match(pred=pred_aspects, gold=gold_aspects)
        TP_A += n_hit_a
        FP_A += (len(pred_aspects) - n_hit_a)
        FN_A += (len(gold_aspects) - n_hit_a)

    precision_a = float(TP_A) / float(TP_A + FP_A + 0.00001)
    recall_a = float(TP_A) / float(TP_A + FN_A + 0.00001)
    F1_a = 2 * precision_a * recall_a / (precision_a + recall_a + 0.00001)

    return precision_a, recall_a, F1_a


def eval_result(dir):
    input_file = os.path.join(dir, 'pre.txt')
    output_file = os.path.join(dir, 'ae_eval_result.txt')
    with open(input_file, 'r', encoding='utf-8') as fp:
        lines = fp.read().splitlines()
        # print('dataset: ', len(lines), input_file)
    dataset = []
    test_Y, pred_Y = [], []
    for l in lines:
        sentence, pre, gold = l.split('***')
        dataset.append({'sentence': sentence, "words": sentence.split()})
        new_pre, new_gold = [], []
        for p_label, g_label in zip(pre.split(), gold.split()):
            if g_label != '-1':  # some token with labels equal -1 will be ignored
                new_pre.append(p_label)
                new_gold.append(g_label)
        test_Y.append(new_gold)
        pred_Y.append(new_pre)

    p, r, f1, output_lines = evaluate_chunk(test_Y, pred_Y, dataset)
    print('Main result for Aspect extract task precision: {} recall: {} f1: {}'.format(p, r, f1))
    with open(output_file, 'w', encoding='utf-8') as fp:
        fp.write('P: ' + str(p) + ' R: ' + str(r) + ' f1: ' + str(f1) + '\n')
        fp.write(' '.join(output_lines))
    return f1

def eval_ts_result(dir):
    input_file = os.path.join(dir, 'pre.txt')
    ads_output_file = os.path.join(dir, 'subtask_ae_eval_result.txt')
    ad_output_file = os.path.join(dir, 'absa_eval_result.txt')
    with open(input_file, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
    dataset = []
    test_Y, pred_Y = [], []
    for l in lines:
        sentence, pre, gold = l.split('***')
        dataset.append({'sentence': sentence, "words": sentence.split()})
        new_pre, new_gold = [], []
        for p_label, g_label in zip(pre.split(), gold.split()):
        	if g_label != '-1':  # some token with labels equal -1 will be ignored
        		new_pre.append(p_label)
        		new_gold.append(g_label)
        test_Y.append(new_gold)
        pred_Y.append(new_pre)

    ad_p, ad_r, ad_f1 = evaluate_chunk(test_Y, pred_Y)
    ads_p, ads_r, ads_f1 = evaluate_absa_chunk(test_Y, pred_Y)
    print('Aspect extract result: precision: {} recall: {} f1: {}'.format(ad_p, ad_r, ad_f1))
    print('Main results for End2end ABSA task： precision: {} recall: {} f1: {}'.format(ads_p, ads_r, ads_f1))
    with open(ad_output_file, 'w', encoding='utf-8') as fp:
        fp.write('P: ' + str(ad_p) + ' R: ' + str(ad_r) + ' f1: ' + str(ad_f1) + '\n')
    with open(ads_output_file, 'w', encoding='utf-8') as fp:
        fp.write('P: ' + str(ads_p) + ' R: ' + str(ads_r) + ' f1: ' + str(ads_f1) + '\n')
    return ads_f1


