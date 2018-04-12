import tensorflow as tf

from typing import List

import numpy as np
import dota.common as common
import dota.encoding as encoding
from model.nn import NNModel, MODEL_PATH


class ModelWrapper(object):
    def __init__(self,
                 pick_encoder: encoding.PickEncoder = encoding.PickEncoder(),
                 game_encoder: encoding.CmPickGameEncoder = encoding.CmPickGameEncoder()):
        self._pick_encoder = pick_encoder
        self._game_encoder = game_encoder

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self._sess = tf.Session(config=config)
        self._nn = NNModel(self._pick_encoder.encoded_shape, self._game_encoder.encoded_shape, len(common.ACTION_SPACE))
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        self._nn.load_if_exists(self._sess)

    def train_picks(self, matches: List[common.Match]):
        picks = self._pick_encoder.encode_multiple([x.pick for x in matches])
        results = np.zeros(shape=[len(matches), 2], dtype=np.float32)
        for i, match in enumerate(matches):
            results[i, match.winning_side] = 1

        return self._nn.train_picks(self._sess, 0.5, picks, results)


    def predict_pick_win(self, pick: common.Pick):
        results = self._nn.predict_win(self._sess, [self._pick_encoder.encode(pick)])
        radiant_probability = results[0]
        return radiant_probability, 1.0 - radiant_probability

    def predict_policy_value(self, game: common.CaptainsModePickGame):
        policy, value = self._nn.predict_policy_value(self._sess, [self._game_encoder.encode(game)])
        policy = policy[0]
        value = value[0]
        return policy, value

    def save(self, path=MODEL_PATH):
        self._nn.save(self._sess, path)

    def __del__(self):
        self._sess.close()
