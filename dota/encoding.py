import numpy as np

from dota.common import *
from dota.heroes import *


def get_hero_index(hero: Hero) -> int:
    return list(Hero).index(hero)


class Encoder(object):
    def __init__(self, encoded_shape: tuple, encoded_dtype=np.float32):
        self.encoded_shape = encoded_shape
        self.encoded_dtype = encoded_dtype

    def encode(self, obj: object) -> np.ndarray:
        raise NotImplementedError

    def encode_multiple(self, collection: List[object]) -> np.ndarray:
        data = np.empty((len(collection), ) + self.encoded_shape, dtype=self.encoded_dtype)
        for i, item in enumerate(collection):
            data[i] = self.encode(item)
        return data

    def decode(self, data: np.ndarray) -> object:
        raise NotImplementedError


class PickEncoder(Encoder):
    def __init__(self):
        super().__init__((NUM_HEROES * 3, ))

    def encode(self, pick: Pick) -> np.ndarray:
        data = np.zeros(self.encoded_shape, dtype=np.float32)
        for picked_hero in pick.radiant:
            data[get_hero_index(picked_hero.hero) * 3] = 1.0
            data[get_hero_index(picked_hero.hero) * 3 + 1] = picked_hero.lane
            data[get_hero_index(picked_hero.hero) * 3 + 2] = picked_hero.role
        for picked_hero in pick.dire:
            data[get_hero_index(picked_hero.hero) * 3] = -1.0
            data[get_hero_index(picked_hero.hero) * 3 + 1] = picked_hero.lane
            data[get_hero_index(picked_hero.hero) * 3 + 2] = picked_hero.role

        return data

    def decode(self, data: np.ndarray) -> Pick:
        # TODO: Implement
        raise NotImplementedError


class CmPickGameEncoder(Encoder):
    def __init__(self):
        # Shape is <Current phase>, <Picked heroes>, <Banned heroes>
        super().__init__((1 + NUM_HEROES + NUM_HEROES, ))

    def encode(self, game: CaptainsModePickGame) -> np.ndarray:
        data = np.zeros(self.encoded_shape, dtype=np.float32)

        # Current phase
        data[0] = game.step

        # Picked heroes
        for picked_hero in game.pick.radiant:
            data[1 + get_hero_index(picked_hero.hero)] = 1.0
        for picked_hero in game.pick.dire:
            data[1 + get_hero_index(picked_hero.hero)] = -1.0

        # Banned heroes
        for hero in game.banned:
            data[1 + NUM_HEROES + get_hero_index(hero)] = 1.0

        return data

    def decode(self, data: np.ndarray) -> CaptainsModePickGame:
        # TODO: Implement
        raise NotImplementedError


def test():
    game = CaptainsModePickGame()
    while not game.is_completed():
        a = np.random.choice(game.get_legal_actions())
        print(a)
        game.do_action(a)
    print(game.pick)
    print(game.banned)

    enc = PickEncoder()
    print(enc.encode(game.pick))

    enc = CmPickGameEncoder()
    print(enc.encode(game))


if __name__ == '__main__':
    test()
