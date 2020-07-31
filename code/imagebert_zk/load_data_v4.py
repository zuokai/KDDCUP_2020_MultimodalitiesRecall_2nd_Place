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

tokenizer = tokenization.FullTokenizer(
        vocab_file='../user_data/vocab.txt', do_lower_case=True)

dict_multimodal_labels = {}
for line in open("../data/multimodal_labels.txt"):
    arr = line.strip().split("\t")
    label = arr[1].replace(","," ").replace("."," ").replace("("," ").replace(")", " ")
    dict_multimodal_labels[arr[0]] = label.strip()

files = tf.gfile.ListDirectory('../data/testB/')

print("read query started!")
dict_querytag_index = {}
dict_label_index = {}
dict_query_index = {}
query_product_list = []
train_query_labels_index = 0
for line in open("../user_data/query_labels.txt"):
    arr = line.strip().split("\t")
    query_tag = arr[1].split(" ")[-1]
    dict_query_index[arr[1].strip()] = ""
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

extra_words = ['letters hooded', 'hooded letters', 'baby high waisted', 'drop resistance cute cup', 'school bag', 'student bag', 'cheongsam']
extra_words.append('flower brooch')
extra_words.append('chandelier')
extra_words.append('handbag')
extra_words.append('hand bag')
extra_words.append('swimsuit')

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

def same_words(query1, query2):
    query1_list = query1.split(" ")
    query2_list = query2.split(" ")
    same_count = 0
    for q1 in query1_list:
        for q2 in query2_list:
            if q1 == q2:
                same_count += 1
    return same_count

def rand_query(query):
   query_list = query.split(" ")
   rand_query_list = query_list
   #print("rand ", rand_query_list)
   rand_val = random.random()
   if len(query_list) <=3:
       return query
   if rand_val < 0.7:
       rand_query_list = query_list
   elif rand_val < 0.8:
       temp = rand_query_list[:-1]
       random.shuffle(temp)
       rand_query_list = temp + [rand_query_list[-1]]
   else:
       temp = rand_query_list[:-2]
       random.shuffle(temp)
       rand_query_list = temp + rand_query_list[-2:]
   return " ".join(rand_query_list)

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
    if FLAGS.sen2forest == 1:
        query = query.replace("sen department of", "forest style")
    #query = rand_query(query)
    len_class_labels = [len(clabel) for clabel in idx_class_labels]
    idx_class_labels = seq_padding(idx_class_labels, 8, 0)
    idx_query = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenizer.tokenize(query) + ['[SEP]'])
    query_id = int(arr[8])
    #e_t = time.time()
    #print("read one line time: ", e_t - s_t)
    return product_id, image_h, image_w, num_boxes, boxes_5, images_features, idx_class_labels, len_class_labels, \
           idx_query, query_id, query, str_class_labels

def read_neg_line(line):
    arr = line.strip().split("\t")
    product_id = int(arr[0])
    query = arr[1]
    query = rand_query(query)
    class_labels = np.array(arr[2].split(","))
    query_tag = query.split(" ")[-1]
    return product_id, query, class_labels, query_tag

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

def generator(file_type = 'train', batch_size = 256):
    rank_label = json.load(open("../data/valid/valid_answer.json"))
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
    query_label_list = []
    weight_query_label_list = []
    segment_type = [0 for i in range(20)] + [1 for i in range(10)]
    epoch_num = 0.0
    while 1:
        random.shuffle(files)
        print("files: ", files)
        epoch_num += 1.0
        neg_ratio = min(epoch_num / 3.0, 1.0)
        print("neg_ratio: ", neg_ratio)
        for filename in files:
            if file_type not in filename:
                continue
            f = open('../data/testB/' + filename, 'r')
            lines = f.readlines()
            dict_tag_lineindex = {}
            lineindex = -1
            for line in lines:
                lineindex += 1
                if "product_id" in line:
                    continue
                query_tag = line.strip().split("\t")[7].split(" ")[-1]
                if query_tag not in dict_tag_lineindex:
                    dict_tag_lineindex[query_tag] = [lineindex]
                else:
                    dict_tag_lineindex[query_tag].append(lineindex)
            #lines = lines[:500]
            index_list = [i for i in range(len(lines))]
            random.shuffle(index_list)
            read_line_time = 0
            all_time = 0
            book_ratio = 0.2
            count = 0
            for index in index_list:
                try:
                    line = lines[index]
                    if "product_id" in line:
                        continue
                    if 'train' in file_type:
                        query = line.strip().split("\t")[7]
                        extra_selected = False
                        for extra_word in extra_words:
                            if extra_word in query:
                                extra_selected = True
                                break
                        if query.strip() not in dict_query_index and extra_selected is False:
                            continue
                    s_t = time.time()
                    product_id, image_h, image_w, num_boxes, boxes_5, images_features, idx_class_labels, len_class_labels, \
                    idx_query, query_id, query, class_label = read_line(line)
                    if "train" in file_type:
                        if "book" in query and random.random() > 0.0:
                            continue
                        if len(idx_query) > 20:
                            continue
                    e_t = time.time()
                    read_line_time += e_t - s_t
                    if 'valid' in file_type:
                        if str(query_id) in rank_label and product_id in rank_label[str(query_id)]:
                            label_list.append(1)
                        else:
                            label_list.append(0)
                    else:
                        label_list.append(1)
                    query_label_list.append([1] * (MAX_LENGTH-2))
                    weight_query_label_list.append([1] * (len(idx_query)-2) + [0] * (MAX_LENGTH-len(idx_query)))
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
                    random_neg = random.random()
                    query_tag = query.split(" ")[-1]
                    search_count = 0
                    all_time += time.time() - e_t
                    search_flag = True
                    while 'train' in file_type:
                        #print("train")
                        index2 = -1
                        search_count += 1
                        if search_count > 10:
                            random_neg = random.random()
                            search_flag = False
                        if search_count > 15:
                            break
                        if random_neg < 0.5 * neg_ratio and query_tag in dict_querytag_index:
                            index2 = random.choice(dict_querytag_index[query_tag])
                        if random_neg >= 0.5 * neg_ratio and random_neg <= 0.7 * neg_ratio:
                            class_label_ = random.choice(class_label)
                            index2 = random.choice(dict_label_index[class_label_])
                        if random_neg > 0.70 * neg_ratio and random_neg <= 0.90 * neg_ratio:
                            b_class_label = filter(lambda x: x != 'others', class_label)
                            if len(b_class_label) != 0:
                                class_label_ = random.choice(b_class_label)
                                index2 = random.choice(dict_label_index[class_label_])
                        if index2 == -1:
                            index2 = random.randint(0, len(query_product_list) - 1)
                        line2 = query_product_list[index2]
                        if "product_id" in line2:
                            print("product_id")
                            continue
                        product_id2, query2, class_labels2, query_tag2 = read_neg_line(line2)
                        if (query.strip() == query2.strip() or product_id == product_id2) and search_flag:
                            continue
                        query_same_count = same_words(query, query2)
                        if (query_same_count == len(query.split(" ")) or query_same_count == len(query2.split(" "))) and search_flag:
                            continue
                        idx_query2 = get_neg_data(query2)
                        if len(idx_query2) > 20 and search_flag:
                            continue
                        elif len(idx_query2) > 20:
                            idx_query2 = idx_query2[:20]

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
                            query_id_list.append(0)
                            len_query_list.append(len(idx_query2))
                            label_list.append(0)
                            segment_type_list.append(segment_type)
                        else:
                            for iii in range(10):
                                if random.random() > 0.5 and query_tag not in dict_tag_lineindex:
                                    line2 = random.choice(lines)
                                else:
                                    line2 = lines[random.choice(dict_tag_lineindex[query_tag])]
                                if "product_id" in line2:
                                    continue
                                product_id2, image_h2, image_w2, num_boxes2, boxes_52, images_features2, idx_class_labels2, len_class_labels2, \
                                             idx_query2, query_id2, query2, class_label2 = read_line(line2)
                                if query.strip() == query2.strip() or product_id == product_id2:
                                    continue
                                break
                            product_id_list.append(product_id2)
                            image_h_list.append(image_h2)
                            image_w_list.append(image_w2)
                            num_boxes_list.append(num_boxes2)
                            boxes_5_list.append(boxes_52)
                            images_features_list.append(images_features2)
                            idx_class_labels_list.append(idx_class_labels2)
                            len_class_labels_list.append(len_class_labels2)
                            idx_query_list.append(idx_query)
                            query_id_list.append(0)
                            len_query_list.append(len(idx_query))
                            label_list.append(0)
                            segment_type_list.append(segment_type)
                        query_labels = [0] * (MAX_LENGTH - 2)
                        query_weights = [0] * (MAX_LENGTH - 2)
                        if len(idx_query2) != 3 and idx_query2[-2] == idx_query[-2]:
                            for query2_index in range(len(idx_query2) - 3):
                                query2_item = idx_query2[query2_index + 1]
                                if query2_item in idx_query:
                                    query_labels[query2_index] = 1
                                query_weights[query2_index] = 1
                        if idx_query2[-2] == idx_query[-2]:
                            query_labels[len(idx_query2) - 3] = 1
                            query_weights[len(idx_query2) - 3] = 1
                        else:
                            query_labels[len(idx_query2) - 3] = 0
                            query_weights[len(idx_query2) - 3] = 1
                        query_label_list.append(query_labels)
                        weight_query_label_list.append(query_weights)
                        break
                    if len(product_id_list) == batch_size:
                        np_boxes_5 = seq_padding_2(boxes_5_list, maxlen=MAX_BOX_NUM, padding_value=0)
                        np_images_features = seq_padding_2(images_features_list, maxlen=MAX_BOX_NUM, padding_value=0)
                        np_idx_class_labels = seq_padding_2(idx_class_labels_list, maxlen=MAX_BOX_NUM, padding_value=0)
                        np_len_class_labels = seq_padding(len_class_labels_list, maxlen=MAX_BOX_NUM, padding_value=0)
                        np_idx_query = seq_padding(idx_query_list, maxlen=MAX_LENGTH, padding_value=0)

                        yield np.array(product_id_list), np.array(image_h_list), np.array(image_w_list), np.array(num_boxes_list), np_boxes_5, \
                               np_images_features, np_idx_class_labels, np_len_class_labels, np_idx_query, \
                               np.array(query_id_list), np.array(label_list), np.array(len_query_list), np.array(segment_type_list), \
                               np.array(query_label_list), np.array(weight_query_label_list)
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
                        query_label_list = []
                        weight_query_label_list = []
                except Exception as e:
                    print("xxxxxxxxxx")
                    import traceback
                    traceback.print_exc()
                    continue
        if "testB.tsv" in file_type:
            break

def generator_val(file_type = 'train', batch_size = 256):
    rank_label = json.load(open("inputs/valid_answer.json"))
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
    query_label_list = []
    weight_query_label_list = []
    segment_type = [0 for i in range(20)] + [1 for i in range(10)]
    epoch_num = 0.0
    while 1:
        #random.shuffle(files)
        print("files: ", files)
        epoch_num += 1.0
        neg_ratio = min(epoch_num / 3.0, 1.0)
        print("neg_ratio: ", neg_ratio)
        for filename in files:
            if file_type not in filename:
                continue
            f = open('inputs/' + filename, 'r')
            lines = f.readlines()
            dict_tag_lineindex = {}
            lineindex = -1
            for line in lines:
                lineindex += 1
                if "product_id" in line:
                    continue
                query_tag = line.strip().split("\t")[7].split(" ")[-1]
                #dict_tag_lineindex[query_tag] = lineindex
                if query_tag not in dict_tag_lineindex:
                    dict_tag_lineindex[query_tag] = [lineindex]
                else:
                    dict_tag_lineindex[query_tag].append(lineindex)
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
                        print("xxx 1")
                        continue
                    if 'train' in file_type:
                        query = line.strip().split("\t")[7]
                        extra_selected = False
                        for extra_word in extra_words:
                            if extra_word in query:
                                extra_selected = True
                                break
                        if query.strip() not in dict_query_index and extra_selected is False:
                            continue
                    s_t = time.time()
                    product_id, image_h, image_w, num_boxes, boxes_5, images_features, idx_class_labels, len_class_labels, \
                    idx_query, query_id, query, class_label = read_line(line)
                    if "book" in query and random.random() > 0.0:
                        continue
                    if len(idx_query) > 20:
                        continue
                    e_t = time.time()
                    read_line_time += e_t - s_t
                    if 'valid' in file_type:
                        if str(query_id) in rank_label and product_id in rank_label[str(query_id)]:
                            label_list.append(1)
                        else:
                            label_list.append(0)
                    else:
                        label_list.append(1)
                    query_label_list.append([1] * (MAX_LENGTH-2))
                    weight_query_label_list.append([1] * (len(idx_query)-2) + [0] * (MAX_LENGTH-len(idx_query)))
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
                    random_neg = random.random()
                    query_tag = query.split(" ")[-1]
                    search_count = 0
                    all_time += time.time() - e_t
                    search_flag = True
                    while 'train' in file_type:
                        #print("train")
                        index2 = -1
                        search_count += 1
                        if search_count > 10:
                            random_neg = random.random()
                            search_flag = False
                        if search_count > 15:
                            break
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
                            print("product_id")
                            continue
                        product_id2, query2, class_labels2, query_tag2 = read_neg_line(line2)
                        if (query.strip() == query2.strip() or product_id == product_id2) and search_flag:
                            continue
                        query_same_count = same_words(query, query2)
                        if (query_same_count == len(query.split(" ")) or query_same_count == len(query2.split(" "))) and search_flag:
                            continue
                        idx_query2 = get_neg_data(query2)

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
                            query_id_list.append(0)
                            len_query_list.append(len(idx_query2))
                            label_list.append(0)
                            segment_type_list.append(segment_type)
                        else:
                            for iii in range(10):
                                if random.random() > 0.5 and query_tag not in dict_tag_lineindex:
                                    line2 = random.choice(lines)
                                else:
                                    line2 = lines[random.choice(dict_tag_lineindex[query_tag])]
                                if "product_id" in line2:
                                    continue
                                product_id2, image_h2, image_w2, num_boxes2, boxes_52, images_features2, idx_class_labels2, len_class_labels2, \
                                             idx_query2, query_id2, query2, class_label2 = read_line(line2)
                                if query.strip() == query2.strip() or product_id == product_id2:
                                    continue
                                break
                            product_id_list.append(product_id2)
                            image_h_list.append(image_h2)
                            image_w_list.append(image_w2)
                            num_boxes_list.append(num_boxes2)
                            boxes_5_list.append(boxes_52)
                            images_features_list.append(images_features2)
                            idx_class_labels_list.append(idx_class_labels2)
                            len_class_labels_list.append(len_class_labels2)
                            idx_query_list.append(idx_query)
                            query_id_list.append(0)
                            len_query_list.append(len(idx_query))
                            label_list.append(0)
                            segment_type_list.append(segment_type)
                        if len(idx_query2) > 20 and search_flag:
                            continue
                        elif len(idx_query2) > 20:
                            idx_query2 = idx_query2[:20]
                        query_labels = [0] * (MAX_LENGTH - 2)
                        query_weights = [0] * (MAX_LENGTH - 2)
                        if len(idx_query2) != 3 and idx_query2[-2] == idx_query[-2]:
                            for query2_index in range(len(idx_query2) - 3):
                                query2_item = idx_query2[query2_index + 1]
                                if query2_item in idx_query:
                                    query_labels[query2_index] = 1
                                query_weights[query2_index] = 1
                        if idx_query2[-2] == idx_query[-2]:
                            query_labels[len(idx_query2) - 3] = 1
                            query_weights[len(idx_query2) - 3] = 1
                        else:
                            query_labels[len(idx_query2) - 3] = 0
                            query_weights[len(idx_query2) - 3] = 1
                        query_label_list.append(query_labels)
                        weight_query_label_list.append(query_weights)
                        break
                    if len(product_id_list) == batch_size:
                        s_t = time.time()
                        np_boxes_5 = seq_padding_2(boxes_5_list, maxlen=MAX_BOX_NUM, padding_value=0)
                        np_images_features = seq_padding_2(images_features_list, maxlen=MAX_BOX_NUM, padding_value=0)
                        np_idx_class_labels = seq_padding_2(idx_class_labels_list, maxlen=MAX_BOX_NUM, padding_value=0)
                        np_len_class_labels = seq_padding(len_class_labels_list, maxlen=MAX_BOX_NUM, padding_value=0)
                        np_idx_query = seq_padding(idx_query_list, maxlen=MAX_LENGTH, padding_value=0)
                        e_t = time.time()
                        all_time = 0.0
                        read_line_time = 0.0

                        yield np.array(product_id_list), np.array(image_h_list), np.array(image_w_list), np.array(num_boxes_list), np_boxes_5, \
                               np_images_features, np_idx_class_labels, np_len_class_labels, np_idx_query, \
                               np.array(query_id_list), np.array(label_list), np.array(len_query_list), np.array(segment_type_list), \
                               np.array(query_label_list), np.array(weight_query_label_list)
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
                        query_label_list = []
                        weight_query_label_list = []
                except Exception as e:
                    print("xxxxxxxxxx")
                    import traceback
                    traceback.print_exc()
                    continue
        if "valid.tsv" in file_type:
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



if __name__ == '__main__':
    while 1:
        next(generator('train', 16))