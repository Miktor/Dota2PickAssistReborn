import json
import numpy as np
import tensorflow as tf
from typing import List

from model.pick_prediction_model import PickPredictionModel
from model.heroes import encode_hero, NUM_HEROES, load_heroes, Hero

PACKED_FILE = 'packed.json'


def read(path=PACKED_FILE):
    with open(path, 'r') as f:
        return json.loads(f.read())


RADIANT = 0
DIRE = 1
BATCH_SIZE = 256


def encode_pick(radiant: List[Hero], dire: List[Hero]):
    pick = np.zeros([2, NUM_HEROES], dtype=np.float32)
    for hero in radiant:
        pick[RADIANT, encode_hero(hero.value)] = 1
    for hero in dire:
        pick[DIRE, encode_hero(hero.value)] = 1
    return pick


def to_training_data(data):
    num_samples = len(data)
    training_heroes = np.zeros(shape=[num_samples, 2, NUM_HEROES], dtype=np.uint8)
    training_results = np.zeros(shape=[num_samples, 2], dtype=np.float32)

    for i, sample in enumerate(data):
        if sample['radiant_win']:
            training_results[i, RADIANT] = 1.0
        else:
            training_results[i, DIRE] = 1.0

        for hero in sample['heroes']:
            if hero['isRadiant']:
                row = RADIANT
            else:
                row = DIRE

            hero_index = encode_hero(hero_id=hero['hero_id'])
            training_heroes[i, row, hero_index] = 1
        assert np.sum(training_heroes[i]) == 10
    return training_heroes, training_results


def main():
    pick1 = encode_pick([Hero.Sven, Hero.Invoker, Hero.ShadowShaman, Hero.Pudge, Hero.Brewmaster],
                       [Hero.PhantomAssassin, Hero.Windranger, Hero.Rubick, Hero.QueenofPain, Hero.Tusk])

    pick2 = encode_pick([Hero.PhantomAssassin, Hero.Windranger, Hero.Rubick, Hero.QueenofPain, Hero.Tusk],
                       [Hero.Sven, Hero.Invoker, Hero.ShadowShaman, Hero.Pudge, Hero.Brewmaster])

    pick3 = encode_pick([Hero.NightStalker, Hero.LegionCommander, Hero.DragonKnight, Hero.PhantomLancer, Hero.ShadowShaman],
                        [Hero.QueenofPain, Hero.Juggernaut, Hero.Earthshaker, Hero.Riki, Hero.SandKing])

    pick4 = encode_pick(
        [Hero.QueenofPain, Hero.Juggernaut, Hero.Earthshaker, Hero.Riki, Hero.SandKing],
        [Hero.NightStalker, Hero.LegionCommander, Hero.DragonKnight, Hero.PhantomLancer, Hero.ShadowShaman])

    picks, results = to_training_data(read())

    model = PickPredictionModel((2, NUM_HEROES), 2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        model.load_if_exists(sess)

        epoch = 0
        while True:
            indices = np.random.randint(0, len(picks), BATCH_SIZE)
            batch_picks = picks[indices]
            batch_results = results[indices]

            loss = model.train(sess, batch_picks, batch_results)
            print('{0} Loss: {1}'.format(epoch, loss))

            if loss < 0.1:
                print('Loss is too small, finished training')
                model.save(sess)
                break

            epoch += 1

        print(model.predict(sess, [pick3, pick4]))


if __name__ == '__main__':
    main()
