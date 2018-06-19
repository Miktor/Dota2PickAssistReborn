from typing import List

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dota.common as common
import dota.encoding as encoding
from model.random_forest import RFModel


class RFWrapper(object):
    def __init__(self,
                 pick_encoder: encoding.PickEncoder = encoding.PickEncoder(),
                 game_encoder: encoding.CmPickGameEncoder = encoding.CmPickGameEncoder()):
        self._pick_encoder = pick_encoder
        self._game_encoder = game_encoder

        self._nn = RFModel(self._pick_encoder.encoded_shape, self._game_encoder.encoded_shape, 0)

    def train_picks(self, matches: List[common.Match], test_matches: List[common.Match]):
        picks = self._pick_encoder.encode_multiple([x.pick for x in matches])
        results = np.zeros(shape=[len(matches), 1], dtype=np.float32)
        for i, match in enumerate(matches):
            results[i, 0] = match.winning_side

        test_picks = self._pick_encoder.encode_multiple([x.pick for x in test_matches])
        test_results = np.zeros(shape=[len(test_matches), 1], dtype=np.float32)
        for i, match in enumerate(test_matches):
            test_results[i, 0] = match.winning_side

        return self._nn.train_picks(picks, results, test_picks, test_results)


    def predict_pick_win(self, pick: common.Pick):
        results = self._nn.predict_win(self._pick_encoder.encode(pick))
        radiant_probability = results[0]
        return radiant_probability, 1.0 - radiant_probability

    def predict_policy_value(self, game: common.CaptainsModePickGame):
        policy, value = self._nn.predict_policy_value([self._game_encoder.encode(game)])
        policy = policy[0]
        value = value[0]
        return policy, value
