# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
import os
import re
import random

import numpy as np
from tqdm import tqdm
from param import args

MAX_LENGTH = 23
MAX_BOX_NUM = 10
MAX_LABLETEXT_LENGTH = 8
KDD_DATA = 'data/kdd'

SHUFFLE_RATIO = args.shuffle_ratio

def read_line(line, dict_multimodal_labels, tokenizer, max_labeltext_length=MAX_LABLETEXT_LENGTH):
    arr = line.strip().split("\t")
    product_id = int(arr[0])
    image_h = int(arr[1])
    image_w = int(arr[2])
    num_boxes = int(arr[3])
    boxes = np.frombuffer(base64.b64decode(
        arr[4]), dtype=np.float32).reshape(num_boxes, 4)
    boxes = boxes / [image_h, image_w, image_h, image_w]
    images_features = np.frombuffer(base64.b64decode(
        arr[5]), dtype=np.float32).reshape(num_boxes, 2048)
    class_labels = np.frombuffer(base64.b64decode(
        arr[6]), dtype=np.int64).reshape(num_boxes)
    if args.shuffle_img:
        boxes, images_features, class_labels = random_img(boxes, images_features, class_labels)

    idx_class_labels = []
    str_class_labels = []
    for class_label in class_labels:
        str_class_labels.append(dict_multimodal_labels[str(class_label)])
        idx_class_labels.append(tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(dict_multimodal_labels[str(class_label)])))

    idx_class_labels, idx_class_labels_mask = seq_padding(
        idx_class_labels, max_labeltext_length, 0)
    query = arr[7]
    if args.shuffle_query:
        query = random_query(query)
    idx_query = tokenizer.convert_tokens_to_ids(
        ['[CLS]'] + tokenizer.tokenize(query) + ['[SEP]'])
    query_id = int(arr[8])
    mask_query, mask_label = random_word(tokenizer.tokenize(query), tokenizer)
    mask_idx_query = tokenizer.convert_tokens_to_ids(
        ['[CLS]'] + mask_query + ['[SEP]'])
    mask_label = [-1] + mask_label + [-1]
    return product_id, boxes, images_features, idx_class_labels, idx_class_labels_mask, \
        idx_query, query_id, query, str_class_labels, mask_query, mask_idx_query, mask_label
    
def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with probability
        ratio = 0.15
        if prob < ratio:
            prob /= ratio

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label

def random_query(query):
    query_list = query.split(" ")
    rand_query_list = query_list
    rand_val = random.random()
    if len(query_list) <=3:
        return query
    if rand_val < SHUFFLE_RATIO:
        rand_query_list = query_list
    elif rand_val < (1+SHUFFLE_RATIO)/2:
        temp = rand_query_list[:-1]
        random.shuffle(temp)
        rand_query_list = temp + [rand_query_list[-1]]
    else:
        temp = rand_query_list[:-2]
        random.shuffle(temp)
        rand_query_list = temp + rand_query_list[-2:]
    return " ".join(rand_query_list)

def random_img(boxes, images_features, class_labels):
    rand_val = random.random()
    if len(boxes) <=3 or rand_val < SHUFFLE_RATIO:
        return boxes, images_features, class_labels
    else:
        index = list(range(len(boxes)))
        random.shuffle(index)
        return boxes[index], images_features[index], class_labels[index]

def seq_padding(X, maxlen=None, padding_value=None, debug=False):
    L = [len(x) for x in X]
    if maxlen is None:
        maxlen = max(L)

    pad_X = np.array([
        np.concatenate([x, [padding_value] * (maxlen - len(x))]) if len(x) < maxlen else x[: maxlen] for x in X
    ])
    pad_mask = np.array([
        np.concatenate([[1] * len(x), [0] * (maxlen - len(x))]) if len(x) < maxlen else [1] * maxlen for x in X
    ])
    if debug:
        print("[!] before pading {}\n".format(X))
        print("[!] after pading {}\n".format(pad_X))
    return pad_X, pad_mask

def seq_padding_2(X, maxlen=None, padding_value=None, debug=False):
    L = [len(x[:, 0]) for x in X]
    if maxlen is None:
        maxlen = max(L)
    col2 = len(X[0][0, :])
    pad_X = np.array([
        np.concatenate([x, padding_value * np.ones(((maxlen - len(x[:, 0])), col2))]) if len(x[:, 0]) < maxlen else x[: maxlen, :] for x in X
    ])
    pad_mask = np.array([
        np.concatenate([[1] * len(x[:, 0]), [0] * (maxlen - len(x[:, 0]))]) if len(x[:, 0]) < maxlen else [1] * maxlen for x in X
    ])
    if debug:
        print("[!] before pading {}\n".format(X))
        print("[!] after pading {}\n".format(pad_X))
    return pad_X, pad_mask

# compute dcg@k for a single sample
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(3, r.size + 2)))
    return 0.

# compute ndcg@k (dcg@k / idcg@k) for a single sample
def get_ndcg(r, ref, k):
    dcg_max = dcg_at_k(ref, k)
    if not dcg_max:
        return 0.
    dcg = dcg_at_k(r, k)
    return dcg / dcg_max

def get_file_count(path, ends='.tsv'):
    ''' 获取指定目录下的所有指定后缀的文件名 '''
    f_list = os.listdir(path)
    count = 0
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == ends:
            count += 1
    return count

def get_file_list(path, ends='.tsv'):
    ''' 获取指定目录下的所有指定后缀的文件名 '''
    f_list = os.listdir(path)
    count = 0
    l = []
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == ends:
            l.append(i)
    return l

def remove_file(path, ends='.tsv'):
    ''' 获取指定目录下的所有指定后缀的文件名 '''
    f_list = os.listdir(path)
    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(i)[1] == ends:
            os.remove(os.path.join(path, i))

def get_label(path):
    # print(os.getcwd())
    remove_list = ['(', ')', '&', 'etc', ',', 'and', '', '/']
    label2id = {}
    id2label = {}
    with open(path) as f:
        lines = f.readlines()
        for line in lines[1:]:
            id = int(line.split('\n')[0].split('\t')[0])
            label = line.split('\n')[0].split('\t')[1]
            label = re.split('[ .,()/]', label)
            for remove_text in remove_list:
                while remove_text in label:
                    label.remove(remove_text)
            label = ' '.join(label)
            label2id[label] = id
            id2label[id] = label
    return label2id, id2label
    