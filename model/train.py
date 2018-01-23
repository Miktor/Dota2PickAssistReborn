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

PACKED_FILE = 'data/packed.json'


def read(path=PACKED_FILE):
    with open(path, 'r') as f:
        return json.loads(f.read())


BATCH_SIZE = 256


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


def test_prediction(sess, model, picks_test, matches_test, results_test):
    sess.run(tf.local_variables_initializer())

    model.update_metrics(sess, picks_test, matches_test, results_test)
    print(model.calc_metrics(sess))


def main():
    picks_raw, results_raw = to_training_data(read())
    x_train, x_test, y_train, y_test = train_test_split(picks_raw, results_raw, train_size=0.8, random_state=13)

    model = PickPredictionModel(input_shape=(MatchEncodeMap.Total,), outputs=ResultsEncodeMap.Total)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M')
        train_writer = tf.summary.FileWriter(os.path.join('C:\\Development\\logs', timestamp),
                                             graph=tf.get_default_graph())
        # model.load_if_exists(sess)

        epoch = 0
        while True:
            indices = np.random.randint(0, len(x_train), BATCH_SIZE)
            loss, summ = model.train(sess, x_train[indices], y_train[indices])
            train_writer.add_summary(summ, epoch)

            if epoch % 1000 == 0 or loss < 0.01:
                print('{0} Loss: {1}'.format(epoch, loss))

            if loss < 0.01:
                print('Loss is {0} @ {1}, finished training'.format(loss, epoch))
                model.save(sess)
                break

            epoch += 1

        test_prediction(sess, model, x_test, y_test)


if __name__ == '__main__':
    main()
