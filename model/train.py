import json
import numpy as np
import tensorflow as tf
from typing import List
from random import shuffle

from model.pick_prediction_model import PickPredictionModel
from model.heroes import encode_hero, NUM_HEROES, load_heroes, Hero

PACKED_FILE = 'data/packed.json'


def read(path=PACKED_FILE):
    with open(path, 'r') as f:
        return json.loads(f.read())


ADVANCED_STATS = False

RADIANT = 0
DIRE = 1
LANE_INDEX = 1

MATCH_DURATION = 0
MATCH_DURATION_DEFAULT = 60 * 45.0  # 45 min as 1.0
PER_MATCH_INPUTS = 1  # duration

LANE_ROW = 2
LANE_ROLE_ROW = 3
ROAM_ROW = 4

BATCH_SIZE = 256
INPUT_PARAMETER_PER_HERO = 1 + 5  # 1-bot, 2 - mid, 3 - top, 4 - RJung, 5 - DJung + 1 + 1  # side, lane, role, roam
MODEL_OUTPUTS = 2  # side


def encode_pick(radiant: List[Hero], dire: List[Hero]):
    pick = np.zeros([NUM_HEROES], dtype=np.float32)
    for hero in radiant:
        pick[encode_hero(hero.value)] = 1.0
    for hero in dire:
        pick[encode_hero(hero.value)] = -1.0
    return pick


def to_training_data(data):
    num_samples = len(data)
    training_heroes = np.zeros(shape=[num_samples, INPUT_PARAMETER_PER_HERO, NUM_HEROES], dtype=np.uint8)
    training_matches = np.zeros(shape=[num_samples, PER_MATCH_INPUTS], dtype=np.float32)
    training_results = np.zeros(shape=[num_samples, MODEL_OUTPUTS], dtype=np.float32)

    for i, sample in enumerate(data):

        training_matches[i, MATCH_DURATION] = sample['duration'] / MATCH_DURATION_DEFAULT

        for hero in sample['heroes']:
            if hero['isRadiant']:
                row = RADIANT
            else:
                row = DIRE

            hero_index = encode_hero(hero_id=hero['hero_id'])
            training_heroes[i, row, hero_index] = 1

            lane = hero['lane']
            training_heroes[i, LANE_INDEX + lane - 1, hero_index] = 1

            if sample['radiant_win']:
                training_results[i, RADIANT] = 1.0
            else:
                training_results[i, DIRE] = 1.0

            if ADVANCED_STATS:
                lane_role = hero['lane_role']
                if hero['is_roaming']:
                    roaming = 1
                else:
                    roaming = 0
                training_results[i, LANE_ROW] = lane
                training_results[i, LANE_ROLE_ROW] = lane_role
                training_results[i, ROAM_ROW] = roaming

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


def main():
    picks, matches, results = to_training_data(read())

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
            indices = np.random.randint(0, len(picks), BATCH_SIZE)
            batch_picks = picks[indices]
            batch_matches = matches[indices]
            batch_results = results[indices]

            loss = model.train(sess, batch_picks, batch_matches, batch_results)
            if epoch % 1000 == 0:
                print('{0} Loss: {1}'.format(epoch, loss))

            if loss < 0.01:
                print('Loss is {0} @ {1}, finished training'.format(loss, epoch))
                # model.save(sess)
                break

            epoch += 1

        test_nn(sess, model)


if __name__ == '__main__':
    main()
