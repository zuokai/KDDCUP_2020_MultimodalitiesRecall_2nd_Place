#/usr/bin/env python
# -*- coding: UTF-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
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
for line in open("multimodal_labels.txt"):
    arr = line.strip().split("\t")
    label = arr[1].replace(","," ").replace("."," ").replace("("," ").replace(")", " ")
    dict_multimodal_labels[arr[0]] = label

files = tf.gfile.ListDirectory('inputs/')

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
    idx_query = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query))
    query_id = int(arr[8])
    return product_id, image_h, image_w, num_boxes, boxes_5, images_features, idx_class_labels, len_class_labels, \
           idx_query, query_id, query, str_class_labels
'''
file_ob = open("valid_answer.txt","w")
file_ob.write("query_id\tproduct_id1\tproduct_id2\tproduct_id3\tproduct_id4\tproduct_id5\tproduct_id6\n")
for line in open("inputs/valid_answer.json"):
    valid_answer = json.loads(line)
    count = 0
    for query_id in valid_answer:
        count = count + 1
        print("count: ", count)
        file_ob.write(query_id + "\t")
        if query_id == "391":
            print "x"
        product_ids = valid_answer[query_id]
        if len(product_ids) >= 6:
            for product_id in product_ids[:5]:
                file_ob.write(str(product_id) + "\t")
            file_ob.write(str(product_ids[-1]) + "\n")
        else:
            for product_id in product_ids:
                file_ob.write(str(product_id) + "\t")
            if len(product_ids) == 5:
                file_ob.write("\n")
            else:
                for i in range(5-len(product_ids)):
                    file_ob.write("\t")
                file_ob.write("\n")
    file_ob.close()
'''
file_ob = open("valid.txt","w")
file_ob.write("product_id\timage_h\timage_w\tnum_boxes\tboxes\tlen_class_labels\tquery_id\tquery\tclass_labels\tdata_src\n")
for line in open('inputs/valid.tsv'):
    try:
        product_id, image_h, image_w, num_boxes, boxes_5, images_features, idx_class_labels, \
        len_class_labels, idx_query, query_id, query, str_class_labels = read_line(line)
        file_ob.write(str(product_id) + "\t")
        file_ob.write(str(image_h) + "\t")
        file_ob.write(str(image_w) + "\t")
        file_ob.write(str(num_boxes) + "\t")
        file_ob.write(str(boxes_5).replace("\n","") + "\t")
        file_ob.write(str(len_class_labels) + "\t")
        file_ob.write(str(query_id) + "\t")
        file_ob.write(str(query) + "\t")
        file_ob.write("train.sample" + "\t")
        file_ob.write(str(str_class_labels) + "\n")

    except:
        continue
file_ob.close()
print "x"

def generator(batch_size = 256):
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
    for filename in files:
        if "train" not in filename:
            continue
        f = open('inputs/' + filename, 'r')
        lines = f.readlines()
        index_list = range(len(lines))
        #random.shuffle(index_list)
        for index in index_list:
                print(index)
            #try:
                line = lines[index]
                product_id, image_h, image_w, num_boxes, boxes_5, images_features, idx_class_labels, len_class_labels, \
                idx_query, query_id = read_line(line)
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
                label_list.append(1)
                while 1:
                    index2 = random.randint(0, len(lines)-1)
                    if index2 == index:
                        continue
                    line2 = lines[index2]
                    product_id2, image_h2, image_w2, num_boxes2, boxes_52, images_features2, idx_class_labels2, len_class_labels2, \
                    idx_query2, query_id2 = read_line(line2)
                    if product_id == product_id2:
                        continue
                    if random.random() < 0.5:
                        product_id_list.append(product_id)
                        image_h_list.append(image_h)
                        image_w_list.append(image_w)
                        num_boxes_list.append(num_boxes)
                        boxes_5_list.append(boxes_5)
                        images_features_list.append(images_features)
                        idx_class_labels_list.append(idx_class_labels)
                        len_class_labels_list.append(len_class_labels)
                        idx_query_list.append(idx_query2)
                        query_id_list.append(query_id2)
                        label_list.append(0)
                    else:
                        product_id_list.append(product_id2)
                        image_h_list.append(image_h2)
                        image_w_list.append(image_w2)
                        num_boxes_list.append(num_boxes2)
                        boxes_5_list.append(boxes_52)
                        images_features_list.append(images_features2)
                        idx_class_labels_list.append(idx_class_labels2)
                        len_class_labels_list.append(len_class_labels2)
                        idx_query_list.append(idx_query)
                        query_id_list.append(query_id)
                        label_list.append(0)
                    break

                if len(product_id_list) == batch_size:
                    np_boxes_5 = seq_padding_2(boxes_5_list, maxlen=MAX_BOX_NUM, padding_value=0)
                    np_images_features = seq_padding_2(images_features_list, maxlen=MAX_BOX_NUM, padding_value=0)
                    np_idx_class_labels = seq_padding_2(idx_class_labels_list, maxlen=MAX_BOX_NUM, padding_value=0)
                    np_len_class_labels = seq_padding(len_class_labels_list, maxlen=MAX_BOX_NUM, padding_value=0)
                    np_idx_query = seq_padding(len_class_labels_list, maxlen=MAX_LENGTH, padding_value=0)

                    yield np.array(product_id_list), np.array(image_h_list), np.array(image_w_list), np.array(num_boxes_list), np_boxes_5, \
                           np_images_features, np_idx_class_labels, np_len_class_labels, np_idx_query, \
                           np.array(query_id_list), np.array(label_list)
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
            #except Exception as e:
            #    import traceback
            #    traceback.print_exc()
            #    continue


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
        next(generator(16))