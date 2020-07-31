import csv
import time
import os
import numpy as np
import base64
import random
import json
from codecs import open

from lxrt.tokenization import BertTokenizer
from utils import read_line, seq_padding, seq_padding_2
from param import args

MAX_LENGTH = 23
MAX_BOX_NUM = 10
MAX_LABLETEXT_LENGTH = 8
KDD_DATA = '../data'


class generator:
    def __init__(self, files, file_type='train-match', batch_size=256):
        self.files = files
        self.file_type = file_type
        self.batch_size = batch_size

        # generate dictionary of labels
        self.dict_multimodal_labels = {}
        for line in open(os.path.join(KDD_DATA, "../data/multimodal_labels.txt"), encoding='utf-8'):
            arr = line.strip().split("\t")
            label = arr[1].replace(",", " ").replace(
                ".", " ").replace("(", " ").replace(")", " ")
            self.dict_multimodal_labels[arr[0]] = label.strip()

        self.tokenizer = BertTokenizer.from_pretrained(
            "../user_data",
            do_lower_case=True
        )

    def get_batch(self):
        files = self.files
        file_type = self.file_type
        batch_size = self.batch_size

        product_id_list = []
        boxes_list = []
        images_features_list = []
        idx_class_labels_list = []
        idx_class_labels_mask_list = []
        idx_query_list = []
        query_id_list = []
        label_list = []
        mask_query_list = []
        mask_idx_query_list = []
        mask_label_list = []

        epoch_num = 0.0
        while 1:
            random.shuffle(files)
            for filename in files:
                with open(os.path.join(KDD_DATA, filename), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                index_list = [i for i in range(len(lines))]
                random.shuffle(index_list)

                for i, index in enumerate(index_list):
                    try:
                        line = lines[index]
                        if "product_id" in line:
                            continue
                        product_id, boxes, images_features, idx_class_labels, idx_class_labels_mask, \
                            idx_query, query_id, query, class_label, mask_query, mask_idx_query, mask_label = read_line(line, self.dict_multimodal_labels, self.tokenizer)

                        label_list.append(1)

                        product_id_list.append(product_id)
                        boxes_list.append(boxes)
                        images_features_list.append(images_features)
                        idx_class_labels_list.append(idx_class_labels)
                        idx_class_labels_mask_list.append(
                            idx_class_labels_mask)
                        idx_query_list.append(idx_query)
                        query_id_list.append(query_id)
                        mask_query_list.append(mask_query)
                        mask_idx_query_list.append(mask_idx_query)
                        mask_label_list.append(mask_label)

                        if len(product_id_list) == batch_size or i == (len(index_list)-1):
                            np_boxes, _ = seq_padding_2(
                                boxes_list, maxlen=MAX_BOX_NUM, padding_value=0)
                            np_images_features, np_images_features_mask = seq_padding_2(
                                images_features_list, maxlen=MAX_BOX_NUM, padding_value=0)
                            np_idx_class_labels, _ = seq_padding_2(
                                idx_class_labels_list, maxlen=MAX_BOX_NUM, padding_value=0)
                            np_idx_class_labels_mask, _ = seq_padding_2(
                                idx_class_labels_mask_list, maxlen=MAX_BOX_NUM, padding_value=0)
                            np_idx_query, np_idx_query_mask = seq_padding(
                                idx_query_list, maxlen=MAX_LENGTH, padding_value=0)
                            np_mask_idx_query, np_mask_idx_query_mask = seq_padding(
                                mask_idx_query_list, maxlen=MAX_LENGTH, padding_value=0)
                            np_mask_label, _ = seq_padding(
                                mask_label_list, maxlen=MAX_LENGTH, padding_value=-1)

                            yield product_id_list, np_boxes, np_images_features, np_images_features_mask, \
                                np_idx_class_labels, np_idx_class_labels_mask, \
                                query_id_list, np_idx_query, np_idx_query_mask, \
                                np_mask_idx_query, np_mask_idx_query_mask, np_mask_label, \
                                np.array(label_list)

                            product_id_list = []
                            boxes_list = []
                            images_features_list = []
                            idx_class_labels_list = []
                            idx_class_labels_mask_list = []
                            idx_query_list = []
                            query_id_list = []
                            label_list = []
                            mask_query_list = []
                            mask_idx_query_list = []
                            mask_label_list = []

                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        continue

    def read_neg_line(self, line):
        arr = line.strip().split("\t")
        product_id = int(arr[0])
        query = arr[1]
        class_labels = np.array(arr[2].split(","))
        query_tag = query.split(" ")[-1]
        return product_id, query, class_labels, query_tag

    def get_neg_data(self, query):
        idx_query = self.tokenizer.convert_tokens_to_ids(
            ['[CLS]'] + self.tokenizer.tokenize(query) + ['[SEP]'])
        return idx_query