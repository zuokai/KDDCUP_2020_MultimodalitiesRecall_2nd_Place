import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import tarfile
import os
#import load_data_v2
import cv2
#import model_triple as model
import os
from tensorflow.python import pywrap_tensorflow
from evaluate_function import evaluate

tf.app.flags.DEFINE_integer('input_size', 224, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 1, '')
tf.app.flags.DEFINE_integer('num_readers', 1, '')
tf.app.flags.DEFINE_float('learning_rate', 0.00002, '')
tf.app.flags.DEFINE_integer('max_steps', 400000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('checkpoint_path', '../prediction_result/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 5000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_integer("sen2forest", 0, "sen2forest")
tf.app.flags.DEFINE_string('pretrained_model_path', '../models/model_attention_kdd_am_word_match_finetune_valid.ckpt-251', '')
flags = tf.app.flags
flags.DEFINE_string("output_root", "", "result output root")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_integer("workers", 20, "work node num")
flags.DEFINE_string("worker_hosts", "20", "work node num")
flags.DEFINE_string("ps_hosts", "", "work node num")
flags.DEFINE_string("job_name", "worker", "job_name")
flags.DEFINE_string("undefork", "", "list of undefined args")
flags.DEFINE_string("log_dir", "", "Index of picurl in input data")
tf.app.flags.DEFINE_string("chief_hosts", "0", "task_index")
tf.app.flags.DEFINE_string("path", "0", "path")
tf.app.flags.DEFINE_string("evaluator_hosts", "0", "task_index")
tf.app.flags.DEFINE_string("url_root", "0", "task_index")
tf.app.flags.DEFINE_integer('batch_size', 256, '')

FLAGS = tf.app.flags.FLAGS
import load_data_v4 as load_data
import model_triple as model
# gpus = list(range(len(FLAGS.gpu_list.split(','))))
gpus = [0]
CLASSES = 10  # the softmax classes
GPU_NUM = 4  # the default gpu num
CLASSES_FOOD = 2
MAX_LENGTH = 20
MAX_BOX_NUM = 10



def tower_loss(num_boxes, np_boxes_5, np_images_features, np_idx_class_labels, np_len_class_labels,
               np_idx_query_, label_list_, len_query_, segment_ids, is_training, label_query, weight_label_query, reuse_variables):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        model_loss, probs, loss_list = model.model_attention_channel_e(num_boxes, np_boxes_5, np_images_features, np_idx_class_labels, np_len_class_labels,
               np_idx_query_, len_query_, label_list_, segment_ids, label_query, weight_label_query, is_training=is_training, reuse=reuse_variables)
    #acc = tf.reduce_mean(tf.cast(probs[:,0] > 0.5, tf.float32))
    preds = tf.argmax(probs, axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(preds, label_list_), dtype=tf.float32))
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    if reuse_variables is None:
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss, acc, probs, loss_list

def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, v in grad_and_vars:
      if g is None:
          print("xx")
      #clip
      g = tf.clip_by_value(g, -1., 1.)
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def open_file(path):
    if not tf.gfile.Exists(path):
        tf.gfile.MkDir(path)
    output_path = path + "testB_result_match_keyword_valid_finetune_251.txt"
    if tf.gfile.Exists(output_path):
        file_mode = 'a'
    else:
        file_mode = 'w'
    file_ob = tf.gfile.GFile(output_path, file_mode)
    return file_ob

def main(argv=None):
    '''
    import os
    files_path = "./inputs/"
    files = []
    #files = os.listdir(files_path)
    for file in files:
        if ".tar.gz" not in file:
            continue
        tar_path = "./" + file
        if not os.path.exists(tar_path):
            os.mkdir(tar_path)
        tar = tarfile.open(files_path + file)
        names = tar.getnames()
        for name in names:
            tar.extract(name, path=tar_path)
        tar.close()
    '''
    num_boxes_list = tf.placeholder(tf.int32, shape=[None,], name='num_boxes_list')
    np_boxes_5 = tf.placeholder(tf.float32, shape=[None, MAX_BOX_NUM, 5], name='np_boxes_5')
    np_images_features = tf.placeholder(tf.float32, shape=[None, MAX_BOX_NUM, 2048], name='np_images_features')
    np_idx_class_labels = tf.placeholder(tf.int32, shape=[None, MAX_BOX_NUM, 8], name='np_idx_class_labels')
    np_len_class_labels = tf.placeholder(tf.int32, shape=[None, MAX_BOX_NUM], name='np_len_class_labels')
    np_idx_query = tf.placeholder(tf.int32, shape=[None, MAX_LENGTH], name='np_idx_query')
    label_list = tf.placeholder(tf.int64, shape=[None,], name='label_list')
    len_query_list = tf.placeholder(tf.int32, shape=[None,], name='len_query_list')
    segment_ids_list = tf.placeholder(tf.int32, shape=[None, MAX_BOX_NUM + MAX_LENGTH], name='segment_ids_list')
    is_training_tensor = tf.placeholder(tf.bool, shape=[], name='is_training_tensor')
    label_querys_list = tf.placeholder(tf.int64, shape=[None, MAX_LENGTH-2], name='label_querys_list')
    weights_label_querys_list = tf.placeholder(tf.float32, shape=[None, MAX_LENGTH-2], name='weights_label_querys_list')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=2500, decay_rate=0.94,
                                               staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

    print("gpus: ", gpus)

    # split
    num_boxes_split = tf.split(num_boxes_list, len(gpus))
    np_boxes_5_split = tf.split(np_boxes_5, len(gpus))
    np_images_features_split = tf.split(np_images_features, len(gpus))
    np_idx_class_labels_split = tf.split(np_idx_class_labels, len(gpus))
    np_len_class_labels_split = tf.split(np_len_class_labels, len(gpus))
    np_idx_query_split = tf.split(np_idx_query, len(gpus))
    label_list_split = tf.split(label_list, len(gpus))
    len_query_split = tf.split(len_query_list, len(gpus))
    segment_ids_split = tf.split(segment_ids_list, len(gpus))
    label_querys_list_split = tf.split(label_querys_list, len(gpus))
    weights_label_querys_list_split = tf.split(weights_label_querys_list, len(gpus))

    tower_grads = []
    reuse_variables = None
    probs_all = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                num_boxes_ = num_boxes_split[i]
                np_boxes_5_ = np_boxes_5_split[i]
                np_images_features_ = np_images_features_split[i]
                np_idx_class_labels_ = np_idx_class_labels_split[i]
                np_len_class_labels_ = np_len_class_labels_split[i]
                np_idx_query_ = np_idx_query_split[i]
                label_list_ = label_list_split[i]
                len_query_ = len_query_split[i]
                segment_ids_ = segment_ids_split[i]
                label_querys_ = label_querys_list_split[i]
                weights_label_querys_ = weights_label_querys_list_split[i]
                total_loss, model_loss, acc, probs, loss_list = tower_loss(num_boxes_, np_boxes_5_, np_images_features_,
                                                         np_idx_class_labels_, np_len_class_labels_,
                                                         np_idx_query_, label_list_, len_query_, segment_ids_,
                                                         is_training_tensor, label_querys_, weights_label_querys_, reuse_variables)
                if probs_all is None:
                    probs_all = probs
                else:
                    probs_all = tf.concat([probs_all, probs], axis=0)
                reuse_variables = True

    variable_averages = tf.train.ExponentialMovingAverage(0.9999)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore, max_to_keep=20)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    file_ob = open_file(FLAGS.checkpoint_path)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        saver.restore(sess, FLAGS.pretrained_model_path)

        data_generator = load_data.get_batch(num_workers=FLAGS.num_readers,
                                         file_type='testB.tsv',
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))
        eval_data_generator = load_data.get_batch(num_workers=FLAGS.num_readers,
                                         file_type='testA.tsv',
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))
        start = time.time()
        dict_eval = {}
        for step in range(FLAGS.max_steps):
          try:
            data = next(data_generator)
            #print len(data)
            #print data
            probs_, tl = sess.run([probs, total_loss], feed_dict={num_boxes_list: data[3],
                                                                              np_boxes_5: data[4],
                                                                              np_images_features: data[5],
                                                                              np_idx_class_labels: data[6],
                                                                              np_len_class_labels: data[7],
                                                                              np_idx_query: data[8],
                                                                              label_list: data[10],
                                                                              len_query_list: data[11],
                                                                              segment_ids_list: data[12],
                                                                              label_querys_list: data[13],
                                                                              weights_label_querys_list: data[14],
                                                                              is_training_tensor: False})

            batch_size = len(data[9])
            for eval_index in range(batch_size):
                file_ob.write(str(data[9][eval_index]) + "\t" + str(data[0][eval_index]) + "\t" + str(
                    probs_[eval_index][1]) + "\n")
                if str(data[9][eval_index]) not in dict_eval:
                    dict_eval[str(data[9][eval_index])] = [(str(data[0][eval_index]), probs_[eval_index][1])]
                else:
                    dict_eval[str(data[9][eval_index])].append(
                        (str(data[0][eval_index]), probs_[eval_index][1]))
            file_ob.flush()
          except:
            print("finished")
            break
if __name__ == '__main__':
    tf.app.run()