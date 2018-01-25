import tensorflow as tf
import os
import numpy as np

MODEL_PATH = './trained-model/pick_prediction'
LEARNING_RATE = 5 * 1e-5
L2_BETA = 0.001

METRICS = {'auc': tf.metrics.auc}


class PickPredictionModel(object):

    def __init__(self, inputs, outputs):
        self.metrics_array = {}
        self.metrics_update_array = []

        with tf.variable_scope('Inputs'):
            self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, inputs])
            self.target_results = tf.placeholder(dtype=tf.float32, shape=[None, outputs])
            self.dropout = tf.placeholder(dtype=tf.float32)

        with tf.variable_scope('Base'):
            flat_input = tf.contrib.layers.flatten(self.inputs)

            hid_1 = tf.contrib.layers.fully_connected(flat_input, 1024, activation_fn=tf.nn.relu)
            hid_1 = tf.contrib.layers.batch_norm(hid_1, center=True, scale=True, is_training=True, scope='bn')
            # hid_1 = tf.nn.dropout(hid_1, keep_prob=self.dropout)

            hid_2 = tf.contrib.layers.fully_connected(hid_1, 1024, activation_fn=tf.nn.relu)
            hid_2 = tf.contrib.layers.batch_norm(hid_2, center=True, scale=True, is_training=True, scope='bn')
            # hid_2 = tf.nn.dropout(hid_2, keep_prob=self.dropout)

            hid_exit = hid_2

        with tf.variable_scope('Head'):
            self.logits = tf.contrib.layers.fully_connected(hid_exit, outputs)
            self.predictions = tf.nn.softmax(self.logits)
            # self.logits = tf.Print(self.logits, [self.logits], message="logits: ")

        with tf.variable_scope('Optimization'):
            with tf.variable_scope('Loss'):
                self.loss = tf.losses.softmax_cross_entropy(self.target_results, self.logits)

            with tf.name_scope('accuracy'):
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(self.logits, 1)), 'float32'))

            with tf.variable_scope('Optimizer'):
                optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
                self.optimize_op = optimizer.minimize(self.loss)

        trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        trainable_weights = [v for v in trainable if 'weights' in v.name]
        for tv in trainable_weights:
            tf.summary.histogram(tv.name, tv)

        with tf.variable_scope('Stats'):
            for metric_name, metric_fn in METRICS.items():
                metric_val, metric_up = metric_fn(predictions=self.predictions, labels=self.target_results)
                self.metrics_array[metric_name] = (metric_val, metric_up)
                tf.summary.scalar(metric_name, metric_val)
                #tf.summary.scalar(metric_name, self.metrics_array[metric_name][0])

            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('cross_entropy_loss', self.loss)
            tf.summary.histogram('logits', self.logits)
            tf.summary.histogram('predictions', self.predictions)

            self.merged_summaries = tf.summary.merge_all()

        with tf.variable_scope('Saver'):
            self.saver = tf.train.Saver()

    def train(self, sess: tf.Session, dropout, inputs, results):
        metric_values_tensors = []
        metric_update_ops = []
        for value_tensor, update_op in self.metrics_array.values():
            metric_values_tensors.append(value_tensor)
            metric_update_ops.append(update_op)

        results = sess.run(
            [self.loss, self.accuracy, self.optimize_op, self.merged_summaries] + metric_values_tensors +
            metric_update_ops,
            feed_dict={
                self.dropout: dropout,
                self.inputs: inputs,
                self.target_results: results
            })

        metric_values = results[4:3 + len(metric_values_tensors)]
        return results[0], results[1], results[2], {
            name: value
            for name, value in zip(self.metrics_array.keys(), metric_values)
        }

    def get_summaries(self, sess: tf.Session):
        summ = sess.run([self.merged_summaries])
        return summ

    def predict(self, sess: tf.Session, inputs):
        return sess.run([self.predictions], feed_dict={self.inputs: inputs})

    def evaluate(self, sess: tf.Session, inputs, target_results):
        metric_values_tensors = []
        metric_update_ops = []
        for value_tensor, update_op in self.metrics_array.values():
            metric_values_tensors.append(value_tensor)
            metric_update_ops.append(update_op)

        results = sess.run(
            metric_values_tensors + metric_update_ops,
            feed_dict={
                self.dropout: 1.0,
                self.inputs: inputs,
                self.target_results: target_results
            })

        metric_values = results[:len(metric_values_tensors)]
        return {name: value for name, value in zip(self.metrics_array.keys(), metric_values)}

    def metrics(self, sess: tf.Session):
        metric_values_tensors = [v for v, _ in self.metrics_array.values()]
        metric_values = sess.run(metric_values_tensors)
        return {name: value for name, value in zip(self.metrics_array.keys(), metric_values)}

    def save(self, sess, path=MODEL_PATH):
        full_path = self.saver.save(sess, path)
        print('Model saved to {0}'.format(full_path))

    def load(self, sess, path=MODEL_PATH):
        self.saver.restore(sess, path)
        print('Model restored')

    def load_if_exists(self, sess, path=MODEL_PATH):
        try:
            if os.path.exists(path + '.meta'):
                print('Model already exists, loading: {0}'.format(path))
                self.load(sess, path)
        except Exception:
            print('Failed to load!')
