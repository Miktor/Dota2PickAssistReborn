import tensorflow as tf
import os

MODEL_PATH = './trained-model/pick_prediction'
LEARNING_RATE = 1e-4
L2_BETA = 0.001


class PickPredictionModel(object):

    def __init__(self, hero_input_shape, match_input_shape, outputs):
        with tf.variable_scope('Inputs'):
            self.picks = tf.placeholder(
                dtype=tf.float32, shape=[
                    None,
                ] + list(hero_input_shape), name='Heroes')

            self.match_details = tf.placeholder(dtype=tf.float32, shape=[None, match_input_shape], name='MatchDetails')
            self.target_results = tf.placeholder(dtype=tf.float32, shape=[None, outputs], name='TargetResults')

        with tf.variable_scope('Base'):
            flat_picks = tf.contrib.layers.flatten(self.picks)
            flat_match_details = tf.contrib.layers.flatten(self.match_details)

            net = tf.concat((flat_picks, flat_match_details), axis=1)

            # net = tf.contrib.layers.flatten([self.picks, self.match_details])

            net = tf.contrib.layers.fully_connected(
                net,
                1024,
                activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_BETA))
            net = tf.contrib.layers.fully_connected(
                net,
                1024,
                activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_BETA))
            net = tf.contrib.layers.fully_connected(
                net, 512, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_BETA))
            net = tf.contrib.layers.fully_connected(
                net, 256, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_BETA))
            net = tf.contrib.layers.fully_connected(
                net, 128, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_BETA))
            net = tf.contrib.layers.fully_connected(
                net, 64, activation_fn=tf.nn.relu, weights_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_BETA))

        with tf.variable_scope('Head'):
            net = tf.contrib.layers.fully_connected(
                net,
                outputs,
                activation_fn=tf.nn.relu,
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=L2_BETA))
            self.predictions = tf.contrib.layers.fully_connected(net, outputs, activation_fn=tf.nn.softmax)

        with tf.variable_scope('Optimization'):
            with tf.variable_scope('Loss'):
                l2_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self.loss = tf.losses.softmax_cross_entropy(self.target_results, net) + l2_loss

            with tf.variable_scope('Optimizer'):
                optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
                self.optimize_op = optimizer.minimize(self.loss)

        with tf.variable_scope('Evaluation'):
            won_side = tf.clip_by_value(self.predictions, -1.0, 1.0)

            with tf.variable_scope('AUC'):
                self.auc, self.update_op_auc = tf.metrics.auc(predictions=won_side, labels=self.target_results)

            with tf.variable_scope('Accuracy'):
                self.accuracy, self.update_op_accuracy = tf.metrics.accuracy(
                    predictions=won_side, labels=self.target_results)

            with tf.variable_scope('Precision'):
                self.precision, self.update_op_precision = tf.metrics.precision(
                    predictions=won_side, labels=self.target_results)

        with tf.variable_scope('Saver'):
            self.saver = tf.train.Saver()

    def train(self, sess: tf.Session, picks, matches, results):
        loss, _ = sess.run(
            [self.loss, self.optimize_op],
            feed_dict={
                self.picks: picks,
                self.match_details: matches,
                self.target_results: results
            })
        return loss

    def predict(self, sess: tf.Session, picks, match_details):
        return sess.run([self.predictions], feed_dict={self.picks: picks, self.match_details: match_details})

    def update_metrics(self, sess: tf.Session, picks, match_details, results):
        sess.run(
            [self.update_op_auc, self.update_op_accuracy, self.update_op_precision],
            feed_dict={
                self.picks: picks,
                self.match_details: match_details,
                self.target_results: results
            })

    def calc_metrics(self, sess: tf.Session):
        return sess.run([self.auc, self.accuracy, self.precision])

    def calc_acc(self, sess: tf.Session):
        return sess.run([self.accuracy])

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
