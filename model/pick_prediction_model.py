import tensorflow as tf
import os
import numpy as np

MODEL_PATH = './trained-model/pick_prediction'
LEARNING_RATE = 1e-4

METRICS = {
    'auc': tf.metrics.auc,
    'accuracy': tf.metrics.accuracy,
    'precision': tf.metrics.precision,
}


class PickPredictionModel(object):

    def __init__(self, input_shape: tuple, outputs: int = 2):
        self.metrics_array = {}

        with tf.variable_scope('Inputs'):
            self.inputs = tf.placeholder(dtype=tf.float32, shape=(None,) + input_shape, name='Input')
            self.target_results = tf.placeholder(dtype=tf.float32, shape=[None, outputs], name='TargetResults')
            self.dropout_rate = tf.placeholder(dtype=tf.float32, name='DropoutRate')

        with tf.variable_scope('Base'):

            flat = tf.layers.flatten(self.inputs)

            net = tf.contrib.layers.fully_connected(
                flat, 1024, activation_fn=tf.nn.relu, biases_initializer=tf.zeros_initializer())
            net = tf.layers.dropout(net, self.dropout_rate)

            net = tf.contrib.layers.fully_connected(
                net, 1024, activation_fn=tf.nn.relu, biases_initializer=tf.zeros_initializer())
            net = tf.layers.dropout(net, self.dropout_rate)

        with tf.variable_scope('Head'):
            self.prediction_logits = tf.contrib.layers.fully_connected(net, outputs, activation_fn=tf.nn.relu)
            self.predictions = tf.nn.softmax(self.prediction_logits)

            self.metrics_array["predictions"] = tf.summary.histogram('predictions', self.predictions)
            self.metrics_array["prediction_logits"] = tf.summary.histogram('prediction_logits', self.prediction_logits)

        with tf.variable_scope('Optimization'):
            with tf.variable_scope('Loss'):
                self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.target_results, logits=self.predictions)

            with tf.variable_scope('Optimizer'):
                optimizer = tf.train.AdamOptimizer(LEARNING_RATE, epsilon=1e-4)
                self.optimize_op = optimizer.minimize(self.loss)

        with tf.variable_scope('Stats'):
            for metric_name, metric_fn in METRICS.items():
                self.metrics_array[metric_name] = metric_fn(
                    predictions=self.prediction_logits, labels=self.target_results)
                #tf.summary.scalar(metric_name, self.metrics_array[metric_name][0])

            with tf.name_scope("Histograms"):
                for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                    self.metrics_array["weight_" + w.name] = tf.summary.histogram("weights" + w.name, w)

            tf.summary.scalar('cross_entropy_loss', self.loss)
            self.merged_summaries = tf.summary.merge_all()

        with tf.variable_scope('Saver'):
            self.saver = tf.train.Saver()

    def train(self, sess: tf.Session, dropout_rate, inputs, results):
        loss, _, summ, pred, pred_log = sess.run(
            [self.loss, self.optimize_op, self.merged_summaries, self.predictions, self.prediction_logits],
            feed_dict={
                self.dropout_rate: dropout_rate,
                self.inputs: inputs,
                self.target_results: results
            })
        return loss, summ, pred, pred_log

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
