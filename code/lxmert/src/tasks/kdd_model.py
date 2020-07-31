# coding=utf-8
import os
import collections
import csv
from codecs import open

import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU, BertPreTrainingHeads

from tasks.kdd_data import generator

from utils import get_file_list

MAX_LENGTH = 23
MAX_BOX_NUM = 10
MAX_LABLETEXT_LENGTH = 8
KDD_DATA = '../data'

class KDD:
    def __init__(self):
        # Model
        self.model = KDDModel()

        # Param
        self.batch_per_epoch = args.batch_per_epoch

        # Load model weights
        # Note: It is different from loading LXMERT pre-trained weights.
        if args.load is not None:
            self.load(args.load)

        # GPU options
        if torch.cuda.is_available():
            if args.multiGPU:
                self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()


    def predict(self, mod='valid', save=False):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        match_pred = []
        match_label = []
        rank_score_pred = collections.defaultdict(list)

        with open(os.path.join(KDD_DATA, '%s/%s.tsv' % (mod, mod)), encoding='utf-8') as f:
            data_num = len(f.readlines())-1
        pbar = tqdm(range(data_num//args.batch_size +
                    (1 if data_num % args.batch_size != 0 else 0)))
        files = ['%s/%s.tsv' % (mod, mod)]
        gen = generator(files, mod, args.batch_size).get_batch()

        with torch.no_grad():
            for i in pbar:
                product_id, boxes, feats, feats_mask, \
                    idx_class_labels, idx_class_labels_mask, \
                    ques_id, idx_query, idx_query_mask, \
                    mask_idx_query, mask_idx_query_mask, mask_label, \
                    target = next(gen)

                boxes = torch.tensor(boxes, dtype=torch.float)
                feats = torch.tensor(feats, dtype=torch.float)
                feats_mask = torch.tensor(feats_mask, dtype=torch.float)
                idx_class_labels = torch.tensor(idx_class_labels, dtype=torch.long)
                idx_class_labels_mask = torch.tensor(idx_class_labels_mask, dtype=torch.long)
                idx_query = torch.tensor(idx_query, dtype=torch.long)
                idx_query_mask = torch.tensor(idx_query_mask, dtype=torch.long)
                mask_idx_query = torch.tensor(mask_idx_query, dtype=torch.long)
                mask_idx_query_mask = torch.tensor(mask_idx_query_mask, dtype=torch.long)
                mask_label = torch.tensor(mask_label, dtype=torch.long)
                target = torch.tensor(target, dtype=torch.long)

                if torch.cuda.is_available():
                    feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                    feats_mask = feats_mask.cuda()
                    idx_class_labels = idx_class_labels.cuda()
                    idx_class_labels_mask = idx_class_labels_mask.cuda()
                    idx_query = idx_query.cuda()
                    idx_query_mask = idx_query_mask.cuda()
                    mask_idx_query = mask_idx_query.cuda()
                    mask_idx_query_mask = mask_idx_query_mask.cuda()
                    mask_label = mask_label.cuda()

                _, _, logit = self.model(idx_query, idx_class_labels,
                                   None, idx_query_mask,
                                   None, idx_class_labels_mask,
                                   feats, boxes, feats_mask)

                score_layer = torch.nn.Softmax(1)
                score = score_layer(logit)

                if torch.cuda.is_available():
                    target = target.cpu()
                    score = score.cpu()

                match_label.extend(target.numpy())
                match_pred.extend(score.numpy())
                for qid, pid, l, label in zip(ques_id, product_id, score.numpy(), target.numpy()):
                    rank_score_pred[qid].append((pid, l[-1]))
                pbar.set_description("Predicting")

            match_pred = [np.argmax(l) for l in match_pred]
            
        if save:
            with open('%s/%s.csv' % (args.result, '%s_score_lxmert' % mod), 'wb') as csvfile:
                fieldnames = ['query-id', 'product-id', 'score']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for key, value in rank_score_pred.items():
                    for item in value:
                        writer.writerow({'query-id': key,
                                        'product-id': item[0],
                                        'score': item[1]
                        })
        return match_pred, match_label, rank_score_pred

    def load(self, path):
        print("Load model from %s" % path)
        if torch.cuda.is_available():
            state_dict=torch.load("%s.pth" % path)
        else:
            state_dict=torch.load("%s.pth" % path, map_location='cpu')

        # Print out the differences of pre-trained and model weights.
        load_keys=set(state_dict.keys())
        model_keys=set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)

class KDDModel(nn.Module):
    def __init__(self):
        super(KDDModel, self).__init__()

        # Build LXRT encoder
        self.lxrt_encoder=LXRTEncoder(
            args,
            mode='lx'
        )
        hid_dim=self.lxrt_encoder.dim
        self.config=self.lxrt_encoder.model.config

        # Image-text heads
        self.logit_fc=nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, 2)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

        # AMSoftmax loss heads
        self.logit_W = torch.nn.Parameter(torch.randn(hid_dim, 2), requires_grad=True)
        nn.init.xavier_normal_(self.logit_W, gain=1)

        # MLM heads
        self.cls=BertPreTrainingHeads(
            self.config, self.lxrt_encoder.model.bert.embeddings.word_embeddings.weight)

    def forward(self, input_ids, boxes_label_input_ids,
                segment_ids, input_mask,
                boxes_label_segment_ids, boxes_label_input_mask,
                feats, boxes, visual_attention_mask):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, 1) The logit of the predict score.
        """

        (lang_output, visn_output), pooled_output=self.lxrt_encoder(input_ids, boxes_label_input_ids,
                             segment_ids, input_mask,
                             boxes_label_segment_ids, boxes_label_input_mask,
                             (feats, boxes), visual_attention_mask)
        lang_prediction_scores, cross_relationship_score=self.cls(
            lang_output, pooled_output)
        
        x_norm = torch.norm(pooled_output, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(pooled_output, x_norm)
        
        if args.task_match and args.task_amsloss:
            w_norm = torch.norm(self.logit_W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            w_norm = torch.div(self.logit_W, w_norm)
            logit = torch.mm(x_norm, w_norm)
        else:
            logit = self.logit_fc(pooled_output)

        return x_norm, lang_prediction_scores, logit
