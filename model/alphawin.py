
import tensorflow as tf
import math


NUM_HEROES = 115
HEROES_INPUT = NUM_HEROES * 2


def inference(picked_heroes):
    # inputs

    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([HEROES_INPUT, 2],
                                stddev=1.0 / math.sqrt(float(HEROES_INPUT))),
            name='win_propability')
        biases = tf.Variable(tf.zeros([2]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(picked_heroes, weights) + biases)


def loss(logits, labels):

    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):

    tf.summary.scalar('loss', loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))
