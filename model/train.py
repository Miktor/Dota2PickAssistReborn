import json
import enum
import numpy as np
import tensorflow as tf
from typing import List
from random import shuffle

from sklearn.model_selection import train_test_split

from model.pick_prediction_model import PickPredictionModel
from model.heroes import encode_hero, load_heroes, Hero
from model.input_data import InpitData, MatchEncodeMap

PACKED_FILE = 'data/packed.json'


def read(path=PACKED_FILE):
    with open(path, 'r') as f:
        return json.loads(f.read())


BATCH_SIZE = 256

MODEL_OUTPUTS = 1  # side
WINNER_INDEX = 0
RADIANT = 1.0
DIRE = -1.0


def to_training_data(data):
    num_samples = len(data)
    training_heroes = np.zeros(shape=[num_samples, MatchEncodeMap.Total], dtype=np.float32)
    training_results = np.zeros(shape=[num_samples, MODEL_OUTPUTS], dtype=np.float32)

    for i, sample in enumerate(data):
        InpitData(sample).encode(training_heroes[i, :])

    return training_heroes, training_results


def test_pick_both_sides(sess, model, pick, duration):
    encoded_pick = encode_pick(pick[0], pick[1])
    picks_to_predict = [encoded_pick, np.flip(encoded_pick, 1)]
    print('radiant - {}'.format(pick[0]))
    print('dire - {}'.format(pick[1]))
    print(model.predict(sess, picks_to_predict, [[duration], [duration]]))


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


def split_data(picks, matches, results):
    random_seed = 13
    picks_train, picks_test, matches_train, matches_test, results_train, results_test = train_test_split(
        picks, matches, results, train_size=0.8, random_state=random_seed)

    return ([picks_train, matches_train, results_train], [picks_test, matches_test, results_test])


def main():

    picks_raw, results_raw = to_training_data(read())
    ([picks_train, matches_train, results_train], [picks_test, matches_test, results_test]) = split_data(
        picks_raw, matches_raw, results_raw)

    model = PickPredictionModel(hero_input_shape=(MatchEncodeMap.Total), outputs=MODEL_OUTPUTS)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    with tf.Session(config=config) as sess:

        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter('C:\\Development\\logs', graph=tf.get_default_graph())

        # model.load_if_exists(sess)

        epoch = 0
        while True:
            indices = np.random.randint(0, len(picks_train), BATCH_SIZE)
            batch_picks = picks_train[indices]
            batch_matches = matches_train[indices]
            batch_results = results_train[indices]

            loss = model.train(sess, batch_picks, batch_matches, batch_results)

            if epoch % 1000 == 0 or loss < 0.01:
                print('{0} Loss: {1}'.format(epoch, loss))

            if loss < 0.01:
                print('Loss is {0} @ {1}, finished training'.format(loss, epoch))
                model.save(sess)
                break

            epoch += 1

        test_prediction(sess, model, picks_test, matches_test, results_test)


if __name__ == '__main__':
    main()
