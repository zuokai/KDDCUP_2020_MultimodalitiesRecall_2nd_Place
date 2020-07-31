import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import tarfile
import os
import load_data_v2
import cv2
import model_triple as model
import os
from tensorflow.python import pywrap_tensorflow
from evaluate_function import evaluate


tf.app.flags.DEFINE_integer('input_size', 224, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 64, '')
tf.app.flags.DEFINE_integer('num_readers', 1, '')
tf.app.flags.DEFINE_float('learning_rate', 0.0001, '')
tf.app.flags.DEFINE_integer('max_steps', 400000, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'models/', '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 500, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_string('pretrained_model_path', 'models/model_attention_kdd.ckpt-41551', '')

FLAGS = tf.app.flags.FLAGS

# gpus = list(range(len(FLAGS.gpu_list.split(','))))
gpus = [0]
CLASSES = 10  # the softmax classes
GPU_NUM = 4  # the default gpu num
CLASSES_FOOD = 2
MAX_LENGTH = 20
MAX_BOX_NUM = 10



def tower_loss(num_boxes, np_boxes_5, np_images_features, np_idx_class_labels, np_len_class_labels,
               np_idx_query_, label_list_, len_query_, reuse_variables):
    # Build inference graph
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        logits, mix_features_embedding = model.model_attention(num_boxes, np_boxes_5, np_images_features, np_idx_class_labels, np_len_class_labels,
               np_idx_query_, len_query_, is_training=True, reuse=reuse_variables)

    label1 = tf.reshape(label_list_, shape=[-1, 1])
    cross_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=label1, logits=logits)
    score_list = tf.nn.sigmoid(logits)
    logits = tf.cast(logits > 0.0, tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(logits, tf.cast(label1, tf.int32)), tf.float32))
    model_loss = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    if reuse_variables is None:
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss, acc, score_list

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

    tower_grads = []
    reuse_variables = None
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
                total_loss, model_loss, acc, logits = tower_loss(num_boxes_, np_boxes_5_, np_images_features_,
                                                         np_idx_class_labels_, np_len_class_labels_,
                                                         np_idx_query_, label_list_, len_query_, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True
                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([apply_gradient_op, batch_norm_updates_op, variables_averages_op]):
        train_op = tf.no_op(name='train_op')
    all_vars = tf.global_variables()
    #var_to_restore = [v for v in all_vars if not 'loss' in v.name]
    #var_to_restore = [v for v in var_to_restore if not 'global' in v.name]
    #var_to_restore = [v for v in var_to_restore if not 'power' in v.name]
    #var_to_restore = [v for v in var_to_restore if not 'Adam' in v.name]
    #var_to_restore = [v for v in var_to_restore if not 'ExponentialMovingAverage' in v.name]
    #var_to_restore = [v for v in var_to_restore if not 'logits' in v.name]
    saver = tf.train.Saver(all_vars, max_to_keep=20)
    #saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    #saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=["resnet"]))
    #summary_writer = tf.summary.FileWriter(FLAGS.checkpoint_path, tf.get_default_graph())

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run(init)
        saver.restore(sess, FLAGS.pretrained_model_path)
        initial_global_step = global_step.assign(50)
        sess.run(initial_global_step)


        data_generator = load_data_v2.get_batch(num_workers=FLAGS.num_readers,
                                         file_type='valid.tsv',
                                         batch_size=FLAGS.batch_size_per_gpu * len(gpus))

        start = time.time()
        dict_eval = {}
        file_ob = open("test_ndcg_result.txt","w")
        try:
            for step in range(FLAGS.max_steps):
                data = next(data_generator)
                ml, tl, acc_, logits_ = sess.run([model_loss, total_loss, acc, logits], feed_dict={num_boxes_list: data[3],
                                                                                  np_boxes_5: data[4],
                                                                                  np_images_features: data[5],
                                                                                  np_idx_class_labels: data[6],
                                                                                  np_len_class_labels: data[7],
                                                                                  np_idx_query: data[8],
                                                                                  label_list: data[10],
                                                                                  len_query_list: data[11],
                                                                                  is_trainig_tensor: })
                if np.isnan(tl):
                    print('Loss diverged, stop training')
                    break

                ac = np.mean(acc_)

                #print(data[9])
                #print(data[0])
                batch_size = len(data[9])
                for index in range(batch_size):
                    file_ob.write(str(data[9][index]) + "\t" + str(data[0][index]) + "\t" + str(logits_[index][0]) + "\n")
                    if data[9][index] not in dict_eval:
                        dict_eval[str(data[9][index])] = [(str(data[0][index]), logits_[index][0])]
                    else:
                        dict_eval[str(data[9][index])].append((str(data[0][index]), logits_[index][0]))

                if step > 13500:
                    s = evaluate({}, {}, dict_eval)
                    print("current step: ", step, " ndcg: ", s)

                if step % 10 == 0:
                    avg_time_per_step = (time.time() - start) / 10
                    avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * len(gpus)) / (time.time() - start)
                    start = time.time()
                    print(logits_[:2])
                    print(
                        'Step {:06d}, model loss {:.4f}, total loss {:.4f}, acc {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                            step, ml, tl, ac, avg_time_per_step, avg_examples_per_second))
                    s = evaluate({}, {}, dict_eval)
                    print("current step: ", step, " ndcg: ", s)
        except:
            print("x")
        s = evaluate({}, {}, dict_eval)
        print("ndcg: ", s)


if __name__ == '__main__':
    tf.app.run()