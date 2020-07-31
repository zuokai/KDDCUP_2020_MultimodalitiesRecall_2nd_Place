# coding=utf-8

import os

import torch
import torch.nn as nn

from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTFeatureExtraction as VisualBertForLXRFeature, VISUAL_CONFIG

MAX_BOXEX_NUM = 10

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer, for_labeltext=False):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (i, sent) in enumerate(sents):
        if for_labeltext:
            sent = sent.split('#')
            feature = []
            for i in range(MAX_BOXEX_NUM):
                if i<len(sent):
                    tokens_a = tokenizer.tokenize(sent[i].strip())

                    # Account for [CLS] and [SEP] with "- 2"
                    if len(tokens_a) > max_seq_length - 2:
                        tokens_a = tokens_a[:(max_seq_length - 2)]
                    
                    # Keep segment id which allows loading BERT-weights.
                    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
                else:
                    token = ["[CLS]"] + ["[SEP]"]

                segment_ids = [0] * len(tokens)

                input_ids = tokenizer.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                padding = [0] * (max_seq_length - len(input_ids))
                input_ids += padding
                input_mask += padding
                segment_ids += padding

                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length

                feature.append(
                        InputFeatures(input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids))
            
            features.append(feature)
                
        else:
            tokens_a = tokenizer.tokenize(sent.strip())

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]
            
            # Keep segment id which allows loading BERT-weights.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            features.append(
                    InputFeatures(input_ids=input_ids,
                                input_mask=input_mask,
                                segment_ids=segment_ids))
    return features


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers


class LXRTEncoder(nn.Module):
    def __init__(self, args, mode='x'):
        super(LXRTEncoder, self).__init__()
        # self.max_seq_length = max_seq_length
        # self.max_label_text_length = max_label_text_length
        set_visual_config(args)

        # Build LXRT Model
        self.model = VisualBertForLXRFeature.from_pretrained(
            "../user_data",
            mode=mode
        )

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return 768

    def forward(self, input_ids, boxes_label_input_ids, 
                segment_ids, input_mask,
                boxes_label_segment_ids, boxes_label_input_mask,
                feats, visual_attention_mask=None):
        output = self.model(input_ids, boxes_label_input_ids, 
                            segment_ids, input_mask,
                            boxes_label_segment_ids, boxes_label_input_mask,
                            visual_feats=feats,
                            visual_attention_mask=visual_attention_mask)
        return output

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        if torch.cuda.is_available():
            state_dict = torch.load("%s.pth" % path)
        else:
            state_dict = torch.load("%s.pth" % path, map_location='cpu')
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
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




