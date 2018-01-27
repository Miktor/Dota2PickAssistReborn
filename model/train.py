import json
import enum
import os
import datetime
import numpy as np
import tensorflow as tf
from typing import List
from random import shuffle

from sklearn.model_selection import train_test_split

from model.pick_prediction_model import PickPredictionModel
from model.heroes import encode_hero, load_heroes, Hero
from model.input_data import InputData, MatchEncodeMap, ResultsEncodeMap
import model.keras_model as k
from keras import backend as back

PACKED_FILE = 'data/packed.json'


def read(path=PACKED_FILE):
    with open(path, 'r') as f:
        return json.loads(f.read())


BATCH_SIZE = 64


def to_training_data(data):
    num_samples = len(data)
    training_heroes = np.zeros(shape=[num_samples, MatchEncodeMap.Total], dtype=np.float32)
    training_results = np.zeros(shape=[num_samples, ResultsEncodeMap.Total], dtype=np.float32)

    for i, sample in enumerate(data):
        InputData(sample).encode(training_heroes[i, :], training_results[i, :])

    return training_heroes, training_results


def test_pick_both_sides(sess: tf.Session, model: PickPredictionModel, pick, duration):
    encoded_pick = encode_pick(pick[0], pick[1])
    picks_to_predict = [encoded_pick, np.flip(encoded_pick, 1)]
    print('radiant - {}'.format(pick[0]))
    print('dire - {}'.format(pick[1]))
    print(model.predict(sess, picks_to_predict))


def test_nn(sess, model):
    pick1 = [[Hero.Sven, Hero.Invoker, Hero.ShadowShaman, Hero.Pudge, Hero.Brewmaster],
             [Hero.PhantomAssassin, Hero.Windranger, Hero.Rubick, Hero.QueenofPain, Hero.Tusk]]

    pick2 = [[Hero.NightStalker, Hero.LegionCommander, Hero.DragonKnight, Hero.PhantomLancer, Hero.ShadowShaman],
             [Hero.QueenofPain, Hero.Juggernaut, Hero.Earthshaker, Hero.Riki, Hero.SandKing]]

    pick3 = [[Hero.QueenofPain, Hero.Juggernaut, Hero.Earthshaker, Hero.Riki, Hero.SandKing],
             [Hero.NightStalker, Hero.LegionCommander, Hero.DragonKnight, Hero.PhantomLancer, Hero.ShadowShaman]]

    test_pick_both_sides(sess, model, pick1, 1.0)
    test_pick_both_sides(sess, model, pick2, 1.0)
    test_pick_both_sides(sess, model, pick3, 1.0)


def test_prediction(sess, model, picks_test, results_test):

    model.update_metrics(sess, picks_test, results_test)
    print(model.calc_metrics(sess))


def main_old():
    picks_raw, results_raw = to_training_data(read())
    x_train, x_test, y_train, y_test = train_test_split(picks_raw, results_raw, train_size=0.8, random_state=13)

    model = PickPredictionModel(inputs=MatchEncodeMap.Total, outputs=ResultsEncodeMap.Total)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.Session(config=config) as sess:

        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M')
        train_writer = tf.summary.FileWriter(
            os.path.join('C:\\Development\\logs', timestamp), graph=tf.get_default_graph())
        # model.load_if_exists(sess)

        epoch = 0
        while True:
            indices = np.random.randint(0, len(x_train), BATCH_SIZE)
            x_epoch = x_train[indices]
            y_epoch = y_train[indices]
            loss, acc, summ = model.train(sess, 0.5, x_epoch, y_epoch)
            # train_writer.add_summary(summ, epoch)
            # print(pred_log)
            # print(pred)

            train_writer.add_summary(summ, epoch)

            if epoch % 100 == 0:
                print('{0} Loss: {1}; Acc: {2}'.format(epoch, loss, acc))
                hist = model.get_histograms(sess)
                train_writer.add_summary(hist, epoch)

            if loss < 0.005:
                print('Loss is {0} @ {1}, finished training'.format(loss, epoch))
                model.save(sess)
                break

            epoch += 1

        print(model.evaluate(sess, x_test, y_test))
        # test_prediction(sess, model, x_test, y_test)


import tensorflow as tf

from keras import backend as K


def main():
    picks_raw, results_raw = to_training_data(read())
    x_train, x_test, y_train, y_test = train_test_split(picks_raw, results_raw, train_size=0.8, random_state=13)

    sess = tf.Session()
    back.set_session(sess)

    model = k.KerasModel(input_shape=(MatchEncodeMap.Total,), outputs=ResultsEncodeMap.Total)

    timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M')
    train_writer = tf.summary.FileWriter(os.path.join('C:\\Development\\logs', timestamp), graph=tf.get_default_graph())

    model.train(x_train, y_train, batch_size=BATCH_SIZE, epochs=100)
    score = model.evaluate(x_test, y_test)
    print('Score: {0}'.format(score))


if __name__ == '__main__':
    main_old()
