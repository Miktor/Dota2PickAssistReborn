import tensorflow as tf
import os
import numpy as np

MODEL_PATH = './trained-model/pick_prediction'
LEARNING_RATE = 1e-4
L2_BETA = 0.001

METRICS = {
    'auc': tf.metrics.auc,
    'accuracy': tf.metrics.accuracy,
    'precision': tf.metrics.precision,
}


class PickPredictionModel(object):

    def __init__(self, inputs, outputs):
        self.metrics_array = {}
        self.metrics_update_array = []

        with tf.variable_scope('Inputs'):
            self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, inputs])
            self.target_results = tf.placeholder(dtype=tf.float32, shape=[None, outputs])
            self.dropout = tf.placeholder(dtype=tf.float32)

        with tf.variable_scope('Base'):
            net = tf.contrib.layers.flatten(self.inputs)
            net = tf.contrib.layers.fully_connected(net, 512, activation_fn=tf.nn.relu)
            net = tf.nn.dropout(net, keep_prob=self.dropout)
            net = tf.contrib.layers.fully_connected(net, 512, activation_fn=tf.nn.relu)
            net = tf.nn.dropout(net, keep_prob=self.dropout)

        with tf.variable_scope('Head'):
            self.prediction_logits = tf.contrib.layers.fully_connected(net, outputs, activation_fn=tf.nn.relu)
            self.predictions = tf.nn.softmax(self.prediction_logits)

        with tf.variable_scope('Optimization'):
            with tf.variable_scope('Loss'):
                self.loss = tf.losses.softmax_cross_entropy(self.target_results, self.prediction_logits)

            with tf.variable_scope('Optimizer'):
                optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
                self.optimize_op = optimizer.minimize(self.loss)

        for tv in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.summary.histogram(tv.name, tv)

        with tf.variable_scope('Stats'):
            for metric_name, metric_fn in METRICS.items():
                metric_val, metric_up = metric_fn(predictions=self.predictions, labels=self.target_results)
                self.metrics_array[metric_name] = (metric_val, metric_up)
                tf.summary.scalar(metric_name, metric_val)
                self.metrics_update_array.append(metric_up)
                #tf.summary.scalar(metric_name, self.metrics_array[metric_name][0])

            tf.summary.scalar('cross_entropy_loss', self.loss)
            tf.summary.histogram('predictions', self.predictions)

            self.merged_summaries = tf.summary.merge_all()

        with tf.variable_scope('Saver'):
            self.saver = tf.train.Saver()

    def train(self, sess: tf.Session, dropout, inputs, results):
        results = sess.run(
            [self.loss, self.optimize_op, self.merged_summaries] + self.metrics_update_array,
            feed_dict={
                self.dropout: dropout,
                self.inputs: inputs,
                self.target_results: results
            })

        return results[0], results[2]

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
