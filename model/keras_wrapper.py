from typing import List

from tensorflow import ConfigProto, Session
import numpy as np
import dota.common as common
import dota.encoding as encoding
from model.keras_nn import KerasModel


class KerasModelWrapper(object):

    def __init__(self,
                 pick_encoder: encoding.PickEncoder = encoding.PickEncoder(),
                 game_encoder: encoding.CmPickGameEncoder = encoding.CmPickGameEncoder()):
        self._pick_encoder = pick_encoder
        self._game_encoder = game_encoder

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4
        self._sess = Session(config=config)

        self._nn = KerasModel(self._sess, self._pick_encoder.encoded_shape, self._game_encoder.encoded_shape, 0)

    def train_picks(self, matches: List[common.Match], test_matches: List[common.Match]):
        picks = self._pick_encoder.encode_multiple([x.pick for x in matches])
        results = np.zeros(shape=[len(matches), 1], dtype=np.float32)
        for i, match in enumerate(matches):
            results[i, 0] = match.winning_side

        test_picks = self._pick_encoder.encode_multiple([x.pick for x in test_matches])
        test_results = np.zeros(shape=[len(test_matches), 1], dtype=np.float32)
        for i, match in enumerate(test_matches):
            test_results[i, 0] = match.winning_side

        return self._nn.train(picks, results, test_picks, test_results)

    def predict_pick_win(self, pick: common.Pick):
        results = self._nn.predict([self._pick_encoder.encode(pick)])
        radiant_probability = results[0]
        return radiant_probability, 1.0 - radiant_probability

    def predict_policy_value(self, game: common.CaptainsModePickGame):
        return


if __name__ == '__main__':
    pass