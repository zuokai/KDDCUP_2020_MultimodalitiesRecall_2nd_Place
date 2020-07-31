# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import pixelmodel
import optimization
import tokenization
import json

import random
import time
import evaluation
import load_data_pred
from tensorflow.contrib import slim


config = tf.ConfigProto(log_device_placement=True, allow_soft_placement = True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3


flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string("vocab_file", '../user_data/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string('TrainFilePath','../dataset','Train Split Path')

flags.DEFINE_string('EvalFilePath','../eval','Evaluate Split Path')

flags.DEFINE_string("ProductIDs_Path", '../ProductIDs.json',
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("valid_answer",'../valid_answer.json','valid answer')

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 20, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 10,
                     "Maximum number of masked LM predictions per sequence.")


# flags.DEFINE_bool(
#     "do_whole_word_mask", False,
#     "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 2,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")

flags.DEFINE_integer("maxboxnum",10,"maxboxnum")

flags.DEFINE_string(
    "bert_config_file", '../user_data/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_bool('random_sample', True, 'Whether To random down sampling')


flags.DEFINE_string(
    "output_dir", '../ModelCheckPointGPUS/',
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", '../models/ImageBertKDD.ckpt-85002',
    "Initial checkpoint (usually from a pre-trained BERT model).")


flags.DEFINE_string("cnn_init_checkpoint",'pretrained_model/resnet_v1_50.ckpt',"Init checkpoint(ResNet)")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_integer("n_gpus",4,"Total GPU numbers")

# flags.DEFINE_integer("train_batch_size_per_gpu", 128, "Total batch size for training.")
flags.DEFINE_integer("train_batch_size_per_gpu", 128, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 128, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 1e-4, "The initial learning rate for Adam.")
"""
3000000/(train_batch_size_per_gpu*n_gpus)*epoch(default=40)
"""
flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 30000, "Number of warmup steps.")

# flags.DEFINE_integer("save_checkpoints_steps",5000 ,
#                      "How often to save the model checkpoint.")
flags.DEFINE_integer("save_checkpoints_steps",5000 ,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("max_eval_steps", 7200, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
def _deduplicate_indexed_slices(values, indices):
  """Sums `values` associated with any non-unique `indices`.
  Args:
    values: A `Tensor` with rank >= 1.
    indices: A one-dimensional integer `Tensor`, indexing into the first
    dimension of `values` (as in an IndexedSlices object).
  Returns:
    A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
    de-duplicated version of `indices` and `summed_values` contains the sum of
    `values` slices associated with each unique index.
  """
  unique_indices, new_index_positions = tf.unique(indices)
  summed_values = tf.unsorted_segment_sum(
    values, new_index_positions,
    tf.shape(unique_indices)[0])
  return summed_values, unique_indices


def average_gradients(tower_grads, batch_size, options):
  # calculate average gradient for each shared variable across all GPUs
  average_grads = []
  count = 0
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    # We need to average the gradients across each GPU.
    count += 1
    g0, v0 = grad_and_vars[0]

    if g0 is None:
      # no gradient for this variable, skip it
      average_grads.append((g0, v0))
      continue

    if isinstance(g0, tf.IndexedSlices):
      # If the gradient is type IndexedSlices then this is a sparse
      #   gradient with attributes indices and values.
      # To average, need to concat them individually then create
      #   a new IndexedSlices object.
      indices = []
      values = []
      for g, v in grad_and_vars:
        indices.append(g.indices)
        values.append(g.values)
      all_indices = tf.concat(indices, 0)
      avg_values = tf.concat(values, 0) / len(grad_and_vars)
      # deduplicate across indices
      av, ai = _deduplicate_indexed_slices(avg_values, all_indices)
      grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

    else:
      # a normal tensor can just do a simple average
      grads = []
      for g, v in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)
        # Append on a 'tower' dimension which we will average over
        grads.append(expanded_g)

      # Average over the 'tower' dimension.
      grad = tf.concat(grads, 0)
      grad = tf.reduce_mean(grad, 0)

    # the Variables are redundant because they are shared
    # across towers. So.. just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)

    average_grads.append(grad_and_var)

  assert len(average_grads) == len(list(zip(*tower_grads)))

  return average_grads


def clip_by_global_norm_summary(t_list, clip_norm, norm_name, variables):
  # wrapper around tf.clip_by_global_norm that also does summary ops of norms

  # compute norms
  # use global_norm with one element to handle IndexedSlices vs dense
  norms = [tf.global_norm([t]) for t in t_list]

  # summary ops before clipping
  summary_ops = []
  for ns, v in zip(norms, variables):
    name = 'norm_pre_clip/' + v.name.replace(":", "_")
    summary_ops.append(tf.summary.scalar(name, ns))

  # clip
  clipped_t_list, tf_norm = tf.clip_by_global_norm(t_list, clip_norm)

  # summary ops after clipping
  norms_post = [tf.global_norm([t]) for t in clipped_t_list]
  for ns, v in zip(norms_post, variables):
    name = 'norm_post_clip/' + v.name.replace(":", "_")
    summary_ops.append(tf.summary.scalar(name, ns))

  summary_ops.append(tf.summary.scalar(norm_name, tf_norm))

  return clipped_t_list, tf_norm, summary_ops


def clip_grads(grads, all_clip_norm_val, do_summaries, global_step):
  # grads = [(grad1, var1), (grad2, var2), ...]
  def _clip_norms(grad_and_vars, val, name):
    # grad_and_vars is a list of (g, v) pairs
    grad_tensors = [g for g, v in grad_and_vars]
    vv = [v for g, v in grad_and_vars]
    scaled_val = val
    if do_summaries:
      clipped_tensors, g_norm, so = clip_by_global_norm_summary(
        grad_tensors, scaled_val, name, vv)
    else:
      so = []
      clipped_tensors, g_norm = tf.clip_by_global_norm(
        grad_tensors, scaled_val)

    ret = []
    for t, (g, v) in zip(clipped_tensors, grad_and_vars):
      ret.append((t, v))

    return ret, so

  ret, summary_ops = _clip_norms(grads, all_clip_norm_val, 'norm_grad')

  assert len(ret) == len(grads)

  return ret, summary_ops

def bertmodel(bert_config,bert_init_checkpoint,learning_rate,num_train_steps,num_warmup_steps,use_one_hot_embeddings,features,ngpus,is_training):

  num_gpu = ngpus if is_training else 1
  #if is_training:
  optimizer = optimization.create_optimizer_mgpu(learning_rate, num_train_steps, num_warmup_steps)
  full_query_id = features['query_id']
  full_product_id = features['product_id']

  # for key in features:features[key] = tf.squeeze(features[key],axis=1)
  input_ids = tf.split(features["input_ids"], num_or_size_splits=num_gpu, axis=0)
  # input_mask = features["input_mask"]
  segment_ids = tf.split(features["segment_ids"], num_or_size_splits=num_gpu, axis=0)
  # masked_lm_positions = features["masked_lm_positions"]
  # masked_lm_ids = features["masked_lm_ids"]
  # masked_lm_weights = features["masked_lm_weights"]
  boxes = tf.split(features['boxes'], num_or_size_splits=num_gpu, axis=0)
  boxfeat = tf.split(features['features'], num_or_size_splits=num_gpu, axis=0)
  labelfeat = tf.split(features["labelfeat"], num_or_size_splits=num_gpu, axis=0)
  query_id = tf.split(features["query_id"], num_or_size_splits=num_gpu, axis=0)
  product_id = tf.split(features["product_id"], num_or_size_splits=num_gpu, axis=0)

  next_sentence_labels = tf.split(features["next_sentence_labels"], num_or_size_splits=num_gpu, axis=0)



  tower_grads = []
  train_perplexity = 0
  next_sentence_gpu = 0
  next_sentence_acc_gpus = 0
  next_sentence_op_gpus = 0
  empty = True
  for gpuid in range(num_gpu,num_gpu+1):
    """
    use gpu:1
    """
    with tf.device('/gpu:%d' % gpuid):
      with tf.name_scope('multigpu%d' % gpuid):
        gpuid -= 1
        model = pixelmodel.BertModel(imgfeat=boxfeat[gpuid],
                                     config=bert_config,
                                     is_training=False,
                                     input_ids=input_ids[gpuid],
                                     label_ids = labelfeat[gpuid],
                                     token_type_ids=segment_ids[gpuid],
                                     use_one_hot_embeddings=use_one_hot_embeddings, random_sample=FLAGS.random_sample)

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs, next_sentence_probs) = get_next_sentence_output(
          bert_config, model.get_pooled_output(), next_sentence_labels[gpuid])
        #total_loss =  next_sentence_loss

        tvars = tf.trainable_variables()


        initialized_variable_names = {}
        scaffold_fn = None
        if bert_init_checkpoint and gpuid==0:
          (assignment_map, initialized_variable_names
           ) = pixelmodel.get_assignment_map_from_checkpoint(tvars, bert_init_checkpoint)
          for var in tvars:
            param_name = var.name[:-2]
            tf.get_variable(
              name=param_name + "/adam_m",
              shape=var.shape.as_list(),
              dtype=tf.float32,
              trainable=False,
              initializer=tf.zeros_initializer())
            tf.get_variable(
              name=param_name + "/adam_v",
              shape=var.shape.as_list(),
              dtype=tf.float32,
              trainable=False,
              initializer=tf.zeros_initializer())

          tf.train.init_from_checkpoint(bert_init_checkpoint, assignment_map)

        tf.get_variable_scope().reuse_variables()

        next_sentence_log_probs = tf.reshape(
          next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
          next_sentence_log_probs, axis=-1, output_type=tf.int64)

        #next_sentence_labels[gpuid] = tf.reshape(next_sentence_labels[gpuid], [-1])
        # print(next_sentence_labels[gpuid])
        # next_sentence_labels_expand = tf.expand_dims(next_sentence_labels[gpuid],-1)
        #next_sentence_accuracy = tf.metrics.accuracy(
        #  labels=next_sentence_labels[gpuid], predictions=next_sentence_predictions)

        #loss = total_loss

        #grads = optimizer.compute_gradients(
        #  loss, var_list=tvars,
        #  aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE
        #)
        #tower_grads.append(grads)
        # keep track of loss across all GPUs
        #train_perplexity += loss
        #next_sentence_gpu += next_sentence_loss
        #next_sentence_op_gpus += next_sentence_accuracy[0]
        #next_sentence_acc_gpus += next_sentence_accuracy[1]
        if empty:
          next_sentence_prob_gpus = next_sentence_probs
          empty = False
        else:next_sentence_prob_gpus = tf.concat((next_sentence_prob_gpus,next_sentence_probs),axis=0)

  if not is_training:return next_sentence_prob_gpus

  global_step = tf.train.get_or_create_global_step()
  new_global_step = global_step + 1

  average_grads = average_gradients(tower_grads, None, None)
  average_grads, norm_summary_ops = clip_grads(average_grads, 1.0, True, global_step)

  train_op = optimizer.apply_gradients(average_grads)
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  #
  # masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
  #                                  [-1, masked_lm_log_probs.shape[-1]])
  # masked_lm_predictions = tf.argmax(
  #   masked_lm_log_probs, axis=-1, output_type=tf.int64)
  # masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
  # masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
  # masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
  # masked_lm_accuracy_val,masked_lm_accuracy_op = tf.metrics.accuracy(
  #   labels=masked_lm_ids,
  #   predictions=masked_lm_predictions,
  #   weights=masked_lm_weights)
  # masked_lm_mean_loss = tf.metrics.mean(
  #   values=masked_lm_example_loss, weights=masked_lm_weights)
  #
  # next_sentence_log_probs = tf.reshape(
  #   next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
  # next_sentence_predictions = tf.argmax(
  #   next_sentence_log_probs, axis=-1, output_type=tf.int64)
  #
  # next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
  # next_sentence_accuracy = tf.metrics.accuracy(
  #   labels=next_sentence_labels, predictions=next_sentence_predictions)


  return train_op,train_perplexity/num_gpu,next_sentence_gpu/num_gpu,next_sentence_op_gpus/num_gpu,next_sentence_acc_gpus/num_gpu,next_sentence_prob_gpus,full_query_id,full_product_id



def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""

  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=pixelmodel.get_activation(bert_config.hidden_act),
          kernel_initializer=pixelmodel.create_initializer(
              bert_config.initializer_range))
      input_tensor = pixelmodel.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=pixelmodel.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    probs = tf.nn.softmax(logits,axis=-1)

    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs,probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = pixelmodel.get_shape_list(sequence_tensor, expected_rank=3)

  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]
  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_offsets = tf.to_int64(flat_offsets)
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor



def main(_):
  batch_size = 5
  modelcheckpoint = '../models/ImageBertKDD.ckpt-85002'

  input_ids = tf.placeholder(dtype=tf.int64,shape=(None,FLAGS.max_seq_length),name="input_ids")
  # input_mask = tf.placeholder(dtype=tf.int64,shape=(None,FLAGS.max_seq_length),name="input_mask")
  segment_ids = tf.placeholder(dtype=tf.int64,shape=(None,FLAGS.max_seq_length),name="segment_ids")
  # masked_lm_positions = tf.placeholder(dtype=tf.int64,shape=(None,FLAGS.max_predictions_per_seq),name="masked_lm_positions")
  # masked_lm_ids = tf.placeholder(dtype=tf.int64,shape=(None,FLAGS.max_predictions_per_seq),name="masked_lm_ids")
  # masked_lm_weights = tf.placeholder(dtype=tf.float32,shape=(None,FLAGS.max_predictions_per_seq),name="masked_lm_weights")
  boxes = tf.placeholder(dtype=tf.float32,shape=(None,FLAGS.maxboxnum,5),name="boxes")
  boxfeat = tf.placeholder(dtype=tf.float32,shape=(None,FLAGS.maxboxnum,2048),name="features")
  labelfeat = tf.placeholder(dtype=tf.int64,shape=(None,FLAGS.maxboxnum,8),name="labelfeat")
  query_id = tf.placeholder(dtype=tf.string,shape=(None,),name="query_id")
  product_id = tf.placeholder(dtype=tf.string,shape=(None,),name="product_id")
  # labels = tf.placeholder(dtype=tf.int64,shape=(None,),name="labels")
  # height = tf.placeholder(dtype=tf.int64, shape=(None,), name="height")
  # width = tf.placeholder(dtype=tf.int64, shape=(None,), name="width")

  next_sentence_labels = tf.placeholder(dtype=tf.int64,shape=(None,),name="next_sentence_labels")


  placeholders = [input_ids,segment_ids,boxes,boxfeat,labelfeat,next_sentence_labels,query_id,product_id]
  features = {}
  for feat in placeholders:
    name = feat.name.split(':')[0]
    features[name] = feat
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = pixelmodel.BertConfig.from_json_file(FLAGS.bert_config_file)

  predict_generator = load_data_pred.generator("test",batch_size)

  next_sentence_prob\
        = bertmodel(bert_config=bert_config,bert_init_checkpoint=FLAGS.init_checkpoint,learning_rate=FLAGS.learning_rate,num_train_steps=FLAGS.num_train_steps,num_warmup_steps=FLAGS.num_warmup_steps,use_one_hot_embeddings=False,features=features,is_training=False,ngpus=1)

  var_list = tf.all_variables()

  saver = tf.train.Saver(var_list, max_to_keep=5)
  ndcgtop5 = [[0,0]]*5
  with tf.Session(config=config) as sess:
    saver.restore(sess, modelcheckpoint)
    result = {}
    try:
      for i in range(int(29005 / batch_size)):
        feedbatch = predict_generator.next()
        pred_dict = {}
        for key in placeholders: pred_dict[key] = feedbatch[key.name.split(':')[0]]
        probilitiy = sess.run((next_sentence_prob), feed_dict=pred_dict)
        querys = feedbatch['query_id']
        products = feedbatch['product_id']
        for (query, product, prob) in zip(querys, products, probilitiy):
          if query not in result: result[query] = []
          result[query].append([str(product), str(prob[1])])
          print("query: ", query, product, prob[1])
    except:
      print("load data finished")
    #sumres = 0
    #json.dump(result,open('testB.json','wb'))
    #for val in result.values():
    #  sumres += len(val)
    #print(sumres)
    #assert sumres == 29005
    with open("../prediction_result/testBscore_imagebert.txt",'wb') as file:
      for key,value in result.items():
        for prodid,score in value:
          line = str(key)+'\t'+str(prodid)+'\t'+str(score)+'\n'
          file.write(line)


if __name__ == "__main__":
  tf.app.run()
