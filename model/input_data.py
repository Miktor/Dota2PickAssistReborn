from model.heroes import encode_hero, NUM_HEROES
import numpy as np
import enum

RADIANT = 1.0
DIRE = -1.0
MATCH_DURATION_DEFAULT = 60 * 45.0  # 45 min as 1.0


class HeroEncodeMap(enum.IntEnum):
    HeroId = 0
    LastHeroId = NUM_HEROES - 1
    Radiant = LastHeroId + 1
    Dire = Radiant + 1
    #HeroLane = Dire + 1
    Total = NUM_HEROES


class MatchEncodeMap(enum.IntEnum):
    # MatchDuration = 0
    HeroStart = 0
    HeroEnd = HeroStart + HeroEncodeMap.Total * 10
    Total = HeroEnd


class ResultsEncodeMap(enum.IntEnum):
    Radiant = 0
    Dire = 1
    Total = 2


PARAMETERS_PER_MATCH = 1
PARAMETERS_PER_HERO = NUM_HEROES


class HeroDetails(object):

    def __init__(self, hero_data):
        self.hero_index = encode_hero(hero_id=hero_data['hero_id'])

        if hero_data['isRadiant']:
            self.side = RADIANT
        else:
            self.side = DIRE

        self.lane = hero_data['lane']
        self.lane_role = hero_data['lane_role']
        self.is_roaming = bool(hero_data['is_roaming'])

    def encode(self, output_data):
        output_data[self.hero_index] = self.side

        # output_data[HeroEncodeMap.HeroLane] = self.lane


class InputData(object):

    def __init__(self, json_data):
        self.duration = json_data['duration'] / MATCH_DURATION_DEFAULT

        self.radiant_win = bool(json_data['radiant_win'])

        self.heroes = [HeroDetails(hero_data) for hero_data in json_data['heroes']]

    def encode(self, output_heroes, output_results):
        if output_heroes.shape[0] != MatchEncodeMap.Total:
            raise RuntimeError('Invalid shape')

        if self.radiant_win:
            output_results[ResultsEncodeMap.Radiant] = 1
        else:
            output_results[ResultsEncodeMap.Dire] = 1

        # output_heroes[MatchEncodeMap.MatchDuration] = self.duration

        for i, hero in enumerate(self.heroes):
            begin_i = MatchEncodeMap.HeroStart + HeroEncodeMap.Total * i
            end_i = begin_i + HeroEncodeMap.Total
            hero.encode(output_heroes[begin_i:end_i])


if __name__ == '__main__':
    pass
