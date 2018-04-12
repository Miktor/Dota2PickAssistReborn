import tensorflow as tf
import os

MODEL_PATH = './trained-model/pick_prediction'
LEARNING_RATE = 5 * 1e-5
L2_BETA = 0.001


class NNModel(object):
    def __init__(self,
                 pick_shape: tuple,
                 pick_game_shape: tuple,
                 pick_game_num_actions: int):
        self.histograms = []

        self.metrics_array = {}
        self.metrics_update_array = []

        with tf.variable_scope('PickPredictionModel'):
            with tf.variable_scope('Inputs'):
                self.picks_inputs = tf.placeholder(dtype=tf.float32, shape=(None, ) + pick_shape)
                self.picks_target_results = tf.placeholder(dtype=tf.float32, shape=(None, 2))
                # self.picks_dropout = tf.placeholder(dtype=tf.float32)

            with tf.variable_scope('Base'):
                flat_input = tf.contrib.layers.flatten(self.picks_inputs)

                hid_1 = tf.contrib.layers.fully_connected(flat_input, 1024, activation_fn=tf.nn.relu)
                hid_1 = tf.contrib.layers.batch_norm(hid_1, center=True, scale=True, is_training=True, scope='bn1')
                # hid_1 = tf.nn.dropout(hid_1, keep_prob=self.dropout)

                hid_2 = tf.contrib.layers.fully_connected(hid_1, 1024, activation_fn=tf.nn.relu)
                hid_2 = tf.contrib.layers.batch_norm(hid_2, center=True, scale=True, is_training=True, scope='bn2')
                # hid_2 = tf.nn.dropout(hid_2, keep_prob=self.dropout)

                hid_exit = hid_2

            with tf.variable_scope('Head'):
                self.picks_logits = tf.contrib.layers.fully_connected(hid_exit, 2)
                self.picks_predictions = tf.nn.softmax(self.picks_logits)

            with tf.variable_scope('Optimization'):
                with tf.variable_scope('Loss'):
                    # Loss:
                    #   Binary classification loss:
                    #       L(p, y) = -y log(p) - (1-y) * log(1-p)
                    #       p - predicted probability (self.probability)
                    #       y - targets
                    self.picks_loss = tf.losses.softmax_cross_entropy(self.picks_target_results, self.picks_logits)

                with tf.name_scope('accuracy'):
                    self.picks_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.picks_target_results, 1), tf.argmax(self.picks_logits, 1)), 'float32'))

                with tf.variable_scope('Optimizer'):
                    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
                    self.picks_optimize_op = optimizer.minimize(self.picks_loss, var_list=tf.trainable_variables('PickPredictionModel'))

            with tf.variable_scope('Stats'):
                tf.summary.scalar('accuracy', self.picks_accuracy)
                tf.summary.scalar('cross_entropy_loss', self.picks_loss)
                auc, auc_op = tf.metrics.auc(predictions=self.picks_predictions, labels=self.picks_target_results)
                tf.summary.scalar('auc', auc)

                self.merged_summaries = tf.summary.merge_all()

            trainable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            trainable_weights = [v for v in trainable if 'weights' in v.name]
            self.histograms = [tf.summary.histogram(tv.name, tv) for tv in trainable_weights]
            # self.histograms += [tf.summary.histogram('logits', self.logits), tf.summary.histogram('predictions', self.predictions)]
            self.histograms = tf.summary.merge(self.histograms)

        with tf.variable_scope('GraphPredictionModel'):
            with tf.variable_scope('Inputs'):
                self.policy_states = tf.placeholder(dtype=tf.float32, shape=(None, ) + pick_game_shape)
                self.policy_target_results = tf.placeholder(dtype=tf.float32, shape=(None, pick_game_num_actions))
                self.policy_dropout = tf.placeholder(dtype=tf.float32)

            with tf.variable_scope('Base'):
                flat_input = tf.contrib.layers.flatten(self.policy_states)

                hid_1 = tf.contrib.layers.fully_connected(flat_input, 1024, activation_fn=tf.nn.relu)
                hid_1 = tf.contrib.layers.batch_norm(hid_1, center=True, scale=True, is_training=True, scope='bn1')
                # hid_1 = tf.nn.dropout(hid_1, keep_prob=self.dropout)

                hid_2 = tf.contrib.layers.fully_connected(hid_1, 1024, activation_fn=tf.nn.relu)
                hid_2 = tf.contrib.layers.batch_norm(hid_2, center=True, scale=True, is_training=True, scope='bn2')
                # hid_2 = tf.nn.dropout(hid_2, keep_prob=self.dropout)

                hid_exit = hid_2

            with tf.variable_scope('PolicyHead'):
                x = tf.contrib.layers.fully_connected(hid_exit, 256, activation_fn=tf.nn.relu)
                self.policy_logits = tf.contrib.layers.fully_connected(x, pick_game_num_actions)
                self.policy_predictions = tf.nn.softmax(self.policy_logits)
                # self.logits = tf.Print(self.logits, [self.logits], message="logits: ")

            with tf.variable_scope('ValueHead'):
                x = tf.contrib.layers.fully_connected(hid_exit, 256, activation_fn=tf.nn.relu)
                self.value = tf.contrib.layers.fully_connected(x, 1, activation_fn=tf.nn.tanh, biases_initializer=None)

            with tf.variable_scope('Optimization'):
                with tf.variable_scope('Loss'):
                    self.policy_loss = tf.losses.softmax_cross_entropy(self.policy_target_results, self.policy_logits)

                with tf.name_scope('accuracy'):
                    self.policy_accuracy = tf.reduce_mean(
                            tf.cast(tf.equal(tf.argmax(self.policy_target_results, 1), tf.argmax(self.policy_logits, 1)), 'float32'))

                with tf.variable_scope('Optimizer'):
                    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
                    self.policy_optimize_op = optimizer.minimize(self.policy_loss)

        with tf.variable_scope('Saver'):
            self.saver = tf.train.Saver()

    def train_picks(self, sess: tf.Session, dropout, inputs, results):
        loss, accuracy, _, merged_summaries = sess.run(
            [self.picks_loss, self.picks_accuracy, self.picks_optimize_op, self.merged_summaries],
            feed_dict={
                # self.picks_dropout: dropout,
                self.picks_inputs: inputs,
                self.picks_target_results: results
            })

        return loss, accuracy, merged_summaries

    def get_histograms(self, sess: tf.Session):
        return sess.run(self.histograms)

    def predict_win(self, sess: tf.Session, inputs):
        return sess.run([self.predicted_side], feed_dict={self.picks_inputs: inputs})

    def predict_policy_value(self, sess: tf.Session, states):
        return sess.run([self.policy_predictions, self.value], feed_dict={self.policy_states: states})

    def evaluate(self, sess: tf.Session, inputs, target_results):
        metric_values_tensors = []
        metric_update_ops = []
        for value_tensor, update_op in self.metrics_array.values():
            metric_values_tensors.append(value_tensor)
            metric_update_ops.append(update_op)

        results = sess.run(
            metric_values_tensors + metric_update_ops,
            feed_dict={
                self.picks_dropout: 1.0,
                self.picks_inputs: inputs,
                self.picks_target_results: target_results
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
        except Exception as e:
            print('Failed to load! {}'.format(e))
