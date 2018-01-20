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


RADIANT = 0
DIRE = 1

LANE_ROW = 2
LANE_ROLE_ROW = 3
ROAM_ROW = 4

BATCH_SIZE = 256
INPUT_PARAMETER_PER_HERO = 2  # 1 + 1 + 1  # side, lane, role, roam
MODEL_OUTPUTS = 2  # side


def encode_pick(radiant: List[Hero], dire: List[Hero]):
    pick = np.zeros([2, NUM_HEROES], dtype=np.float32)
    for hero in radiant:
        pick[RADIANT, encode_hero(hero.value)] = 1
    for hero in dire:
        pick[DIRE, encode_hero(hero.value)] = 1
    return pick


def to_training_data(data):
    num_samples = len(data)
    training_heroes = np.zeros(shape=[num_samples, INPUT_PARAMETER_PER_HERO, NUM_HEROES], dtype=np.uint8)
    training_results = np.zeros(shape=[num_samples, MODEL_OUTPUTS], dtype=np.float32)

    for i, sample in enumerate(data):
        for hero in sample['heroes']:
            lane = hero['lane']
            lane_role = hero['lane_role']
            if hero['is_roaming']:
                roaming = 1
            else:
                roaming = 0

            if hero['isRadiant']:
                row = RADIANT
            else:
                row = DIRE

            if sample['radiant_win']:
                training_results[i, RADIANT] = 1.0
            else:
                training_results[i, DIRE] = 1.0

            # training_results[i, LANE_ROW] = lane
            # training_results[i, LANE_ROLE_ROW] = lane_role
            # training_results[i, ROAM_ROW] = roaming

            hero_index = encode_hero(hero_id=hero['hero_id'])
            training_heroes[i, row, hero_index] = 1
        assert np.sum(training_heroes[i]) == 10
    return training_heroes, training_results


def test_pick_both_sides(sess, model, pick):
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

    test_pick_both_sides(sess, model, pick1)
    test_pick_both_sides(sess, model, pick2)
    test_pick_both_sides(sess, model, pick3)


def main():
    picks, results = to_training_data(read())

    model = PickPredictionModel(input_shape=(INPUT_PARAMETER_PER_HERO, NUM_HEROES), outputs=MODEL_OUTPUTS)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        train_writer = tf.summary.FileWriter("C:\Development\logs", graph=tf.get_default_graph())

        # model.load_if_exists(sess)

        epoch = 0
        while True:
            indices = np.random.randint(0, len(picks), BATCH_SIZE)
            batch_picks = picks[indices]
            batch_results = results[indices]

            loss = model.train(sess, batch_picks, batch_results)
            print('{0} Loss: {1}'.format(epoch, loss))

            if loss < 0.1:
                print('Loss is too small, finished training')
                # model.save(sess)
                break

            epoch += 1

        test_nn(sess, model)


if __name__ == '__main__':
    main()
