from typing import List

import numpy as np
from dota.common import Match
from dota.encoding import PickEncoder, CmPickGameEncoder
from dota.json import matches_from_json_file
from sklearn.datasets import dump_svmlight_file

PACKED_FILE = 'data\\packed.json'

class Exporter(object):
    def __init__(self,
                 pick_encoder: PickEncoder = PickEncoder(),
                 game_encoder: CmPickGameEncoder = CmPickGameEncoder()):
        self._pick_encoder = pick_encoder
        self._game_encoder = game_encoder

    def convert(self, matches: List[Match]):
        picks = self._pick_encoder.encode_multiple([x.pick for x in matches])
        results = np.zeros(shape=[len(matches), 1], dtype=np.float32)
        for i, match in enumerate(matches):
            results[i, 0] = match.winning_side
        return picks, results


if __name__ == '__main__':
    matches = matches_from_json_file(PACKED_FILE)
    export = Exporter()
    picks, results = export.convert(matches)
    results = np.reshape(results, (results.shape[0], ))
    dump_svmlight_file(picks, results, 'data\\learn.txt')