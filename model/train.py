import json
import enum
import numpy as np
import tensorflow as tf
from typing import List
from random import shuffle

from sklearn.model_selection import train_test_split

from model.pick_prediction_model import PickPredictionModel
from model.heroes import encode_hero, NUM_HEROES, load_heroes, Hero

PACKED_FILE = 'data/packed.json'


def read(path=PACKED_FILE):
    with open(path, 'r') as f:
        return json.loads(f.read())


BATCH_SIZE = 256
ADVANCED_STATS = False

MATCH_DURATION = 0
MATCH_DURATION_DEFAULT = 60 * 45.0  # 45 min as 1.0
PER_MATCH_INPUTS = 1  # duration

INPUT_PARAMETER_PER_HERO = 1 + 5 + 1  #+ 5  # 1-bot, 2 - mid, 3 - top, 4 - RJung, 5 - DJung + is_roam + role  # side, lane, role, roam
SIDE_INDEX = 0
LANE_INDEX = 1
ROAM_INDEX = 6
ROLE_INDEX = 7

MODEL_OUTPUTS = 1  # side
WINNER_INDEX = 0
RADIANT = 1.0
DIRE = -1.0


class Lane(enum.Enum):
    Bot = 1
    Mid = 2
    Top = 3
    RadiantForest = 4
    DireForest = 5


class Role(enum.Enum):
    Carry = 1
    Support = 2
    Offlane = 3
    RadiantForest = 4
    DireForest = 5


def encode_pick(radiant: List[Hero], dire: List[Hero]):
    pick = np.zeros([NUM_HEROES], dtype=np.float32)
    for hero in radiant:
        pick[encode_hero(hero.value)] = 1.0
    for hero in dire:
        pick[encode_hero(hero.value)] = -1.0
    return pick


def to_training_data(data):
    num_samples = len(data)
    training_heroes = np.zeros(shape=[num_samples, INPUT_PARAMETER_PER_HERO, NUM_HEROES], dtype=np.float32)
    training_matches = np.zeros(shape=[num_samples, PER_MATCH_INPUTS], dtype=np.float32)
    training_results = np.zeros(shape=[num_samples, MODEL_OUTPUTS], dtype=np.float32)

    for i, sample in enumerate(data):

        training_matches[i, MATCH_DURATION] = sample['duration'] / MATCH_DURATION_DEFAULT

        if sample['radiant_win']:
            training_results[i, WINNER_INDEX] = RADIANT
        else:
            training_results[i, WINNER_INDEX] = DIRE

        for hero in sample['heroes']:

            hero_index = encode_hero(hero_id=hero['hero_id'])

            if hero['isRadiant']:
                training_heroes[i, SIDE_INDEX, hero_index] = RADIANT
            else:
                training_heroes[i, SIDE_INDEX, hero_index] = DIRE

            lane = hero['lane']
            training_heroes[i, LANE_INDEX + lane - 1, hero_index] = 1

            if hero['is_roaming']:
                training_heroes[i, ROAM_INDEX] = 1

    return training_heroes, training_matches, training_results


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

    print(model.calc_auc(sess, picks_test, matches_test, results_test))


def split_data(picks, matches, results):
    random_seed = 13
    picks_train, picks_test, matches_train, matches_test, results_train, results_test = train_test_split(
        picks, matches, results, train_size=0.8, random_state=random_seed)

    return ([picks_train, matches_train, results_train], [picks_test, matches_test, results_test])


def main():

    picks_raw, matches_raw, results_raw = to_training_data(read())
    ([picks_train, matches_train, results_train], [picks_test, matches_test, results_test]) = split_data(
        picks_raw, matches_raw, results_raw)

    model = PickPredictionModel(
        hero_input_shape=(INPUT_PARAMETER_PER_HERO, NUM_HEROES),
        match_input_shape=PER_MATCH_INPUTS,
        outputs=MODEL_OUTPUTS)

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
            if epoch % 1000 == 0:
                print('{0} Loss: {1}'.format(epoch, loss))

            if loss < 0.01:
                print('Loss is {0} @ {1}, finished training'.format(loss, epoch))
                model.save(sess)
                break

            epoch += 1

        test_prediction(sess, model, picks_test, matches_test, results_test)


if __name__ == '__main__':
    main()
