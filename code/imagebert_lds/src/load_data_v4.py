#/usr/bin/env python
# -*- coding: UTF-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')
import glob
import csv
import cv2
import time
import os
import numpy as np
import base64
import pickle
import scipy.optimize
# import matplotlib.pyplot as plt
# import matplotlib.patches as Patches
import random
import collections
from math import *
import re
import tensorflow as tf
from data_util import GeneratorEnqueuer
import tokenization
import json

tf.app.flags.DEFINE_string('dir_path', './',
                           'training dataset to use')
FLAGS = tf.app.flags.FLAGS
MAX_LENGTH = 20
MAX_BOX_NUM = 10
MASK_PROB = 0.15
MAX_PRED = 10

tokenizer = tokenization.FullTokenizer(
        vocab_file='../user_data/vocab.txt', do_lower_case=True)

dict_multimodal_labels = {}
for line in open("../multimodal_labels.txt"):
    arr = line.strip().split("\t")
    label = arr[1].replace(","," ").replace("."," ").replace("("," ").replace(")", " ")
    dict_multimodal_labels[arr[0]] = label.strip()

files = tf.gfile.ListDirectory('../dataset/')

print("read query started!")
dict_querytag_index = {}
dict_label_index = {}
query_product_list = []
train_query_labels_index = 0
for line in open("../query_labels.txt"):
    arr = line.strip().split("\t")
    query_tag = arr[1].split(" ")[-1]
    if query_tag not in dict_querytag_index:
        dict_querytag_index[query_tag] = [train_query_labels_index]
    else:
        dict_querytag_index[query_tag].append(train_query_labels_index)
    label_list = arr[2].split(",")
    set_label = set()
    for label in label_list:
        label = label.strip()
        if label in set_label:
            continue
        set_label.add(label)
        if label not in dict_label_index:
            dict_label_index[label] = [train_query_labels_index]
        else:
            dict_label_index[label].append(train_query_labels_index)
    query_product_list.append(line.strip())
    train_query_labels_index += 1
print("read query finished!")

def seq_padding(X, maxlen=None, padding_value=None, debug=False):
    L = [len(x) for x in X]
    if maxlen is None:
        maxlen = max(L)

    pad_X = np.array([
        np.concatenate([x, [padding_value] * (maxlen - len(x))]) if len(x) < maxlen else x[: maxlen] for x in X
    ])
    if debug:
        print("[!] before pading {}\n".format(X))
        print("[!] after pading {}\n".format(pad_X))
    return pad_X

def seq_padding_2(X, maxlen=None, padding_value=None, debug=False):
    L = [len(x[:,0]) for x in X]
    if maxlen is None:
        maxlen = max(L)
    col2 = len(X[0][0,:])
    pad_X = np.array([
        np.concatenate([x, padding_value * np.ones(((maxlen - len(x[:,0])), col2))]) if len(x[:,0]) < maxlen else x[: maxlen,:] for x in X
    ])
    if debug:
        print("[!] before pading {}\n".format(X))
        print("[!] after pading {}\n".format(pad_X))
    return pad_X

def read_line(line):
    #s_t = time.time()
    arr = line.strip().split("\t")
    product_id = int(arr[0])
    image_h = int(arr[1])
    image_w = int(arr[2])
    num_boxes = int(arr[3])
    boxes = np.frombuffer(base64.b64decode(arr[4]), dtype=np.float32).reshape(num_boxes, 4)
    boxes_5 = np.zeros((num_boxes, 5), dtype=np.float32)
    boxes_5[:, :4] = boxes / [image_h, image_w, image_h, image_w]
    boxes_5[:, 4] = (boxes[:, 2] - boxes[:, 0]) * (
                boxes[:, 3] - boxes[:, 1]) / (image_w * image_h)
    images_features = np.frombuffer(base64.b64decode(arr[5]), dtype=np.float32).reshape(num_boxes, 2048)
    class_labels = np.frombuffer(base64.b64decode(arr[6]), dtype=np.int64).reshape(num_boxes)
    str_class_labels = []
    idx_class_labels = []
    for class_label in class_labels:
        str_class_labels.append(dict_multimodal_labels[str(class_label)])
        idx_class_labels.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dict_multimodal_labels[str(class_label)])))
    query = arr[7]
    len_class_labels = [len(clabel) for clabel in idx_class_labels]
    idx_class_labels = seq_padding(idx_class_labels, 8, 0)
    tokens = ['[CLS]'] + tokenizer.tokenize(query) + ['[SEP]']
    idx_query = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(idx_query)
    query_id = int(arr[8])
    #e_t = time.time()
    #print("read one line time: ", e_t - s_t)
    return product_id, image_h, image_w, num_boxes, boxes_5, images_features, idx_class_labels, len_class_labels, \
           idx_query, query_id, query, str_class_labels,tokens,input_mask

def read_neg_line(line):
    arr = line.strip().split("\t")
    product_id = int(arr[0])
    query = arr[1]
    class_labels = np.array(arr[2].split(","))
    query_tag = query.split(" ")[-1]
    tokens = ['[CLS]'] + tokenizer.tokenize(query) + ['[SEP]']
    return product_id, query, class_labels, query_tag , tokens

def has_same_label(class_labels1, class_labels2, remove_other):
    for label in class_labels1:
        for label2 in class_labels2:
            if remove_other is True and (label == 'other' or label2 == 'other'):
                continue
            if label == label2:
                return True
    return False

def get_neg_data(query):
    idx_query = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(query) + ['[SEP]'])
    return idx_query

def tokens_to_masked(tokens,vocab_words,rng):
	(tokens, masked_lm_positions,
	 masked_lm_labels) = create_masked_lm_predictions(tokens, MASK_PROB, MAX_PRED, vocab_words, rng)
	masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
	masked_lm_weights = [1.0] * len(masked_lm_ids)
	return masked_lm_positions,masked_lm_ids,masked_lm_weights


def generator(file_type = 'train', batch_size = 256):
    rng = random.Random()
    vocab_words = list(tokenizer.vocab.keys())
    rank_label = json.load(open("../valid_answer.json"))
    product_id_list = []
    image_h_list = []
    image_w_list = []
    num_boxes_list = []
    boxes_5_list = []
    images_features_list = []
    idx_class_labels_list = []
    len_class_labels_list = []
    idx_query_list = []
    len_query_list = []
    query_id_list = []
    label_list = []
    segment_type_list = []
    segment_type = [0 for i in range(20)]
    epoch_num = 0.0
    masked_lm_weights_list = []
    masked_lm_positions_list = []
    masked_lm_ids_list = []
    input_mask_list = []

    while 1:
        random.shuffle(files)
        print("files: ", files)
        epoch_num += 1.0
        neg_ratio = min(epoch_num / 8.0, 1.0)
        print("neg_ratio: ", neg_ratio)
        for filename in files:
            if file_type not in filename:
                continue
            f = open('../dataset/' + filename, 'r')
            lines = f.readlines()
            #lines = lines[:500]
            index_list = [i for i in range(len(lines))]
            random.shuffle(index_list)
            read_line_time = 0
            all_time = 0
            book_ratio = 0.2
            count = 0
            for index in index_list:
                try:
                    #count += 1
                    #if count > 64:
                    #    break
                    line = lines[index]
                    if "product_id" in line:
                        continue
                    s_t = time.time()
                    product_id, image_h, image_w, num_boxes, boxes_5, images_features, idx_class_labels, len_class_labels, \
                    idx_query, query_id, query, class_label,tokens,input_mask = read_line(line)
                    if "book" in query and random.random() > 0.2:
                        continue
                    e_t = time.time()
                    #print("read_line cost time: ", e_t - s_t)
                    masked_lm_positions, masked_lm_ids, masked_lm_weights = tokens_to_masked(tokens,vocab_words,rng)

                    read_line_time += e_t - s_t
                    if 'valid' in file_type:
                        if str(query_id) in rank_label and product_id in rank_label[str(query_id)]:
                            label_list.append(1)
                        else:
                            label_list.append(0)
                    else:
                        label_list.append(1)
                    product_id_list.append(product_id)
                    image_h_list.append(image_h)
                    image_w_list.append(image_w)
                    num_boxes_list.append(num_boxes)
                    boxes_5_list.append(boxes_5)
                    images_features_list.append(images_features)
                    idx_class_labels_list.append(idx_class_labels)
                    len_class_labels_list.append(len_class_labels)
                    idx_query_list.append(idx_query)
                    query_id_list.append(query_id)
                    len_query_list.append(len(idx_query))
                    segment_type_list.append(segment_type)
                    masked_lm_positions_list.append(masked_lm_positions)
                    masked_lm_ids_list.append(masked_lm_ids)
                    masked_lm_weights_list.append(masked_lm_weights)
                    input_mask_list.append(input_mask)
                    random_neg = random.random()
                    query_tag = query.split(" ")[-1]
                    search_count = 0
                    while 'train' in file_type:
                        #print("train")
                        index2 = -1
                        search_count += 1
                        if search_count > 10:
                            random_neg = random.random()
                        if random_neg < 0.5 * neg_ratio and query_tag in dict_querytag_index:
                            index2 = random.choice(dict_querytag_index[query_tag])
                        if random_neg >= 0.5 * neg_ratio and random_neg <= 0.7 * neg_ratio:
                            class_label_ = random.choice(class_label)
                            #if class_label_ not in dict_label_index:
                            #    continue
                            index2 = random.choice(dict_label_index[class_label_])
                        if random_neg > 0.70 * neg_ratio and random_neg <= 0.90 * neg_ratio:
                            b_class_label = filter(lambda x: x != 'others', class_label)
                            if len(b_class_label) != 0:
                                class_label_ = random.choice(b_class_label)
                                index2 = random.choice(dict_label_index[class_label_])
                        if index2 == -1:
                            index2 = random.randint(0, len(query_product_list) - 1)
                        #index2 = random.randint(0, len(query_product_list) - 1)
                        line2 = query_product_list[index2]
                        if "product_id" in line2:
                            continue
                        product_id2, query2, class_labels2, query_tag2,tokens2 = read_neg_line(line2)
                        masked_lm_positions2, masked_lm_ids2, masked_lm_weights2 = tokens_to_masked(tokens2, vocab_words,
                                                                                                 rng)
                        if query.strip() == query2.strip() or product_id == product_id2:
                            continue
                        idx_query2 = get_neg_data(query2)
                        input_mask2 = [1] * len(idx_query2)

                        if random.random() < 2.0:
                            product_id_list.append(product_id)
                            image_h_list.append(image_h)
                            image_w_list.append(image_w)
                            num_boxes_list.append(num_boxes)
                            boxes_5_list.append(boxes_5)
                            images_features_list.append(images_features)
                            idx_class_labels_list.append(idx_class_labels)
                            len_class_labels_list.append(len_class_labels)
                            idx_query_list.append(idx_query2)
                            masked_lm_weights_list.append(masked_lm_weights2)
                            masked_lm_positions_list.append(masked_lm_positions2)
                            masked_lm_ids_list.append(masked_lm_ids2)
                            input_mask_list.append(input_mask2)
                            query_id_list.append(0)
                            len_query_list.append(len(idx_query2))
                            label_list.append(0)
                            segment_type_list.append(segment_type)
                        break
                    all_time += time.time() - e_t
                    if len(product_id_list) == batch_size:
                        s_t = time.time()
                        np_boxes_5 = seq_padding_2(boxes_5_list, maxlen=MAX_BOX_NUM, padding_value=0)
                        np_images_features = seq_padding_2(images_features_list, maxlen=MAX_BOX_NUM, padding_value=0)
                        np_idx_class_labels = seq_padding_2(idx_class_labels_list, maxlen=MAX_BOX_NUM, padding_value=0)
                        np_len_class_labels = seq_padding(len_class_labels_list, maxlen=MAX_BOX_NUM, padding_value=0)
                        np_idx_query = seq_padding(idx_query_list, maxlen=MAX_LENGTH, padding_value=0)
                        np_mlp_list = seq_padding(masked_lm_positions_list, maxlen=MAX_PRED, padding_value=0)
                        np_mli_list = seq_padding(masked_lm_ids_list, maxlen=MAX_PRED, padding_value=0)
                        np_mlw_list = seq_padding(masked_lm_weights_list, maxlen=MAX_PRED, padding_value=0.0)
                        np_inputmask_list = seq_padding(input_mask_list,maxlen=MAX_LENGTH,padding_value=0)
                        e_t = time.time()
                        #print("padding time: ", e_t - s_t)
                        #print("all_time: ", all_time)
                        #print("read_line time: ", read_line_time)
                        all_time = 0.0
                        read_line_time = 0.0
                        result = {
                            "product_id":np.array([str(id) for id in product_id_list]),
                            "height":np.array([int(h) for h in image_h_list]),
                            "width":np.array([int(w) for w in image_w_list]),
                            "numbox":np.array(num_boxes_list),
                            "boxes":np_boxes_5,
                            "features":np_images_features,

                            "labelfeat":np_idx_class_labels,
                            "boxlabellength":np_len_class_labels,

                            "input_ids":np_idx_query,
                            "query_id":np.array([str(id) for id in query_id_list]),
                            "query_list":np.array(len_query_list),
                            "segment_ids":np.array(segment_type_list),
	                          "masked_lm_positions":np.array(np_mlp_list),
	                          "masked_lm_ids":np.array(np_mli_list),
	                          "masked_lm_weights":np.array(np_mlw_list),
	                          "input_mask":np.array(np_inputmask_list),

                            "next_sentence_labels": np.array(label_list)


                        }
                        yield result

                        # yield np.array(product_id_list), np.array(image_h_list), np.array(image_w_list), np.array(num_boxes_list), np_boxes_5, \
                        #        np_images_features, np_idx_class_labels, np_len_class_labels, np_idx_query, \
                        #        np.array(query_id_list), np.array(label_list), np.array(len_query_list), np.array(segment_type_list)
                        #print("yield data")
                        product_id_list = []
                        image_h_list = []
                        image_w_list = []
                        num_boxes_list = []
                        boxes_5_list = []
                        images_features_list = []
                        idx_class_labels_list = []
                        len_class_labels_list = []
                        idx_query_list = []
                        query_id_list = []
                        label_list = []
                        len_query_list = []
                        segment_type_list = []
                        masked_lm_weights_list = []
                        masked_lm_positions_list = []
                        masked_lm_ids_list = []
                        input_mask_list = []
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    continue
        if "train" not in file_type:
            continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        enqueuer.start(max_queue_size=1, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    do_whole_word_mask = False
    if (do_whole_word_mask and len(cand_indexes) >= 1 and
        token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
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
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)


if __name__ == '__main__':
    gen = generator('train',2)
    for key,value in gen.next().items():
        print(key)
        print(value)
