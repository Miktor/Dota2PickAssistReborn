import tensorflow as tf
import os

MODEL_PATH = './trained-model/pick_prediction'
LEARNING_RATE = 1e-4
L2_BETA = 0.001


class PickPredictionModel(object):
    def __init__(self, input_shape, outputs):
        with tf.variable_scope('Inputs'):
            self.picks = tf.placeholder(
                dtype=tf.float32, shape=[
                    None,
                ] + list(input_shape))
            self.target_results = tf.placeholder(dtype=tf.float32, shape=[None, outputs])

        with tf.variable_scope('Base'):
            net = tf.contrib.layers.flatten(self.picks)
            net = tf.contrib.layers.fully_connected(
                net,
                512,
                activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_BETA))
            net = tf.contrib.layers.fully_connected(
                net,
                512,
                activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_BETA))

        with tf.variable_scope('Head'):
            net = tf.contrib.layers.fully_connected(
                net,
                outputs,
                activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_BETA))
            self.predictions = tf.contrib.layers.fully_connected(net, outputs, activation_fn=tf.nn.softmax)

        with tf.variable_scope('Optimization'):
            with tf.variable_scope('Loss'):
                self.loss = tf.losses.softmax_cross_entropy(self.target_results, net)

            with tf.variable_scope('Optimizer'):
                optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
                self.optimize_op = optimizer.minimize(self.loss)

        with tf.variable_scope('Saver'):
            self.saver = tf.train.Saver()

    def train(self, sess: tf.Session, picks, results):
        loss, _ = sess.run([self.loss, self.optimize_op], feed_dict={self.picks: picks, self.target_results: results})
        return loss

    def predict(self, sess: tf.Session, picks):
        return sess.run([self.predictions], feed_dict={self.picks: picks})

    def save(self, sess, path=MODEL_PATH):
        full_path = self.saver.save(sess, path)
        print('Model saved to {0}'.format(full_path))

    def load(self, sess, path=MODEL_PATH):
        self.saver.restore(sess, path)
        print('Model restored')

    def load_if_exists(self, sess, path=MODEL_PATH):
        if os.path.exists(path + '.meta'):
            print('Model already exists, loading: {0}'.format(path))
            self.load(sess, path)
