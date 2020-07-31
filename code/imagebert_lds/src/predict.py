import load_data_v3
import tensorflow as tf
from run_pretraining_multigpu import bertmodel

def predict(predgenerator,modelfilepath,placeholders,batch_size=30):
	saver = tf.train.Saver()
	result = []
	train_op, total_loss, next_sentence_loss, next_sentence_accuracy, next_sentence_accuracy_op, next_sentence_prob, query_id, product_id \
		= bertmodel(bert_config=bert_config, bert_init_checkpoint=FLAGS.init_checkpoint, learning_rate=FLAGS.learning_rate,
		            num_train_steps=FLAGS.num_train_steps, num_warmup_steps=FLAGS.num_warmup_steps,
		            use_one_hot_embeddings=False, features=features, is_training=False, ngpus=1)
	with tf.Session() as sess:
		saver.restore(sess,modelfilepath)
		for i in range(int(28830/batch_size)):
			feedbatch = predgenerator.next()
			pred_dict = {}
			for key in placeholders: pred_dict[key] = pred_dict[key.name.split(':')[0]]
			probilitiy = sess.run((next_sentence_prob), feed_dict=pred_dict)
			querys = feedbatch['query_id']
			products = feedbatch['product_id']
			for (query, product, prob) in zip(querys, products, probilitiy):
				if query not in result: result[query] = []
				result[query].append([product, prob[1]])
	return result

