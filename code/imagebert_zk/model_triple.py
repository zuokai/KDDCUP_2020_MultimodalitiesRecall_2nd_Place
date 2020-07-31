#/usr/bin/env python
# -*- coding: UTF-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
import os
import io
import pixelbert
import gc

bert_config = pixelbert.BertConfig.from_json_file('../user_data/bert_config.json')

def image_bert(image_features, text_ids, is_training, input_mask, segment_ids):
    model = pixelbert.BertModel(imgfeat=image_features,
                                config=bert_config,
                                is_training=is_training,
                                input_ids=text_ids,
                                input_mask=input_mask,
                                token_type_ids=segment_ids,
                                use_one_hot_embeddings=False)
    return model

def get_next_sentence_output(input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=pixelbert.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    #logits = tf.clip_by_value(logits, -10.0, 10.0)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    probs = tf.nn.softmax(logits,axis=-1)

    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return loss, probs

def amsoftmax_loss(y_true, y_pred):
    scale = 30.0
    margin = 0.35

    y_pred = tf.nn.l2_normalize(y_pred, dim=1)

    kernel = tf.get_variable(name='am_kernel', dtype=tf.float32, shape=[768, 2],
                             initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    kernel_norm = tf.nn.l2_normalize(kernel, 0, 1e-10, name='kernel_norm')
    y_pred = tf.matmul(y_pred, kernel_norm)
    y_pred = tf.clip_by_value(y_pred, -1, 1)

    label = tf.reshape(tf.argmax(y_true, axis=-1), shape=(-1, 1))
    label = tf.cast(label, dtype=tf.int32)  # y
    batch_range = tf.reshape(tf.range(tf.shape(y_pred)[0]), shape=(-1, 1))  # 0~batchsize-1
    indices_of_groundtruth = tf.concat([batch_range, tf.reshape(label, shape=(-1, 1))],
                                       axis=1)  # 2columns vector, 0~batchsize-1 and label
    groundtruth_score = tf.gather_nd(y_pred, indices_of_groundtruth)  # score of groundtruth

    m = tf.constant(margin, name='m')
    s = tf.constant(scale, name='s')

    added_margin = tf.cast(tf.greater(groundtruth_score, m),
                           dtype=tf.float32) * m  # if groundtruth_score>m, groundtruth_score-m
    added_margin = tf.reshape(added_margin, shape=(-1, 1))
    added_embeddingFeature = tf.subtract(y_pred, y_true * added_margin) * s  # s(cos_theta_yi-m), s(cos_theta_j)

    cross_ent = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=added_embeddingFeature)
    #loss = tf.reduce_mean(cross_ent)
    probs = tf.nn.softmax(added_embeddingFeature, axis=-1)
    return cross_ent, probs

def get_next_sentence_output_am(input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):


    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    one_hot_labels = tf.reshape(one_hot_labels, (-1,2))
    loss, probs = amsoftmax_loss(one_hot_labels, input_tensor)
    #w0 = tf.cast(probs[:, 0] > 0.9, tf.float32)
    #w1 = tf.cast(probs[:, 1] > 0.9, tf.float32)
    #weights = 1 - w0 * tf.cast(labels, tf.float32) - w1 * tf.cast((1 - labels), tf.float32)
    loss = tf.reduce_mean(loss)
    #per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    #loss = tf.reduce_mean(per_example_loss)
    return loss, probs

def get_query_match_output_(input_tensor, labels, weights, index):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("kdd/seq_query_match"):
    output_weights = tf.get_variable(
        "output_weights" + str(index),
        shape=[2, bert_config.hidden_size],
        initializer=pixelbert.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias" + str(index), shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    #logits = tf.clip_by_value(logits, -10.0, 10.0)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    probs = tf.nn.softmax(logits, axis=-1)[:, 1] * (2 * weights - 1)
    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1) * weights
    loss = tf.reduce_mean(per_example_loss)
    return loss, probs

def get_query_match_output(input_tensor, image_tensor, labels, weights, index):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("kdd/seq_query_match", reuse=tf.AUTO_REUSE):
    output_weights = tf.get_variable(
        "output_weights" + str(index),
        shape=[2, bert_config.hidden_size],
        initializer=pixelbert.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias" + str(index), shape=[2], initializer=tf.zeros_initializer())

    input_tensor = tf.contrib.layers.fully_connected(input_tensor, 768, None, scope='kdd_query_dense1')
    image_tensor = tf.contrib.layers.fully_connected(image_tensor, 768, None, scope='kdd_query_dense2')
    #outputs = tf.matmul(input_tensor, image_tensor, transpose_b=True)
    #outputs = outputs / (image_tensor.get_shape().as_list()[-1] ** 0.5)
    input_tensor = (input_tensor + image_tensor)
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    #logits = tf.clip_by_value(logits, -10.0, 10.0)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    probs = tf.nn.softmax(logits, axis=-1)

    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1) * weights
    loss = tf.reduce_mean(per_example_loss)
    return loss, probs

def model_attention_channel_e(num_boxes, np_boxes_5, np_images_features, np_idx_class_labels, np_len_class_labels,
                    np_idx_query_, len_query_, labels, segment_ids, label_query, weight_label_query, is_training=True, reuse=None):
    '''
    define the model, we use slim's implemention of resnet
    INPUT:
    tagnames: [x1,x2,x3....]    [,1,10,768]
    predict: [p1,p2,p3,p4,p5]   [,1,5,768]
    '''
    #_embedding = tf.get_variable(name="kdd_W", shape=shape_word_embedding,
    #                initializer=tf.constant_initializer(word_embedding_matrix),
    #                trainable=True)
    #labels_embedding = tf.nn.embedding_lookup(_embedding, tf.cast(np_idx_class_labels, tf.int64),
    #                                       name='labels_embedding')
    #querys_embedding = tf.nn.embedding_lookup(_embedding, tf.cast(np_idx_query_, tf.int64),
    #                                       name='querys_embedding')

    with tf.variable_scope("bert", None):
        with tf.variable_scope("embeddings"):
            labels_embedding = pixelbert.embedding_lookup_label(
                input_ids=np_idx_class_labels,
                vocab_size=bert_config.vocab_size,
                embedding_size=bert_config.hidden_size,
                initializer_range=bert_config.initializer_range,
                word_embedding_name="word_embeddings",
                use_one_hot_embeddings=False)

    with slim.arg_scope([slim.batch_norm, slim.dropout], weight_decay=1e-5, is_training=is_training):
        labels_embedding = slim.conv2d(labels_embedding, 768, [1, 8], stride=1, scope='kdd_conv1')
        labels_embedding = tf.reduce_mean(labels_embedding, axis=2)
        box_embedding = tf.contrib.layers.fully_connected(np_boxes_5, 768, None, scope='kdd_dense1')
        np_images_features = tf.expand_dims(np_images_features, axis=2)
        np_images_features = slim.conv2d(np_images_features, 768, [1, 1], stride=1, scope='kdd_conv2')
        np_images_features = tf.reshape(np_images_features, (-1, 10, 768))
        np_images_features = labels_embedding + box_embedding + np_images_features


    query_mask = tf.sequence_mask(len_query_, 20)
    box_mask = tf.sequence_mask(num_boxes, 10)
    input_mask = tf.concat([query_mask, box_mask], axis=-1)
    input_mask = tf.cast(input_mask, tf.int32)
    loss_list = []
    model = image_bert(np_images_features, np_idx_query_, is_training, input_mask, segment_ids)
    loss, probs = get_next_sentence_output_am(model.get_pooled_output(), labels)
    loss_list.append(loss)
    #query_outputs = model.get_query_output()
    #probs_all = probs[:, 1:2]
    #for i in range(18):
    #    loss1, probs1 = get_query_match_output(query_outputs[i], label_query[:, i], weight_label_query[:, i], i)
    #    loss += loss1
    #    loss_list.append(loss)
    #    probs_all = tf.concat([probs_all, tf.reshape(probs1,(-1,1))], axis=1)

    return loss, probs, loss_list