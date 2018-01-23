from model.heroes import encode_hero, NUM_HEROES
import numpy as np
import enum

RADIANT = 1.0
DIRE = -1.0
MATCH_DURATION_DEFAULT = 60 * 45.0  # 45 min as 1.0


class HeroEncodeMap(enum.IntEnum):
    HeroId = 0
    LastHeroId = NUM_HEROES - 1
    HeroLane = LastHeroId + 1
    HeroRole = HeroLane + 1
    Total = HeroRole + 1


class MatchEncodeMap(enum.IntEnum):
    MatchDuration = 0
    HeroStart = 1
    HeroEnd = HeroStart + HeroEncodeMap.Total * 10 - 1
    Total = HeroEnd + 1


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
        output_data[HeroEncodeMap.HeroLane] = self.lane
        output_data[HeroEncodeMap.HeroRole] = self.lane


class InputData(object):

    def __init__(self, json_data):
        self.duration = json_data['duration'] / MATCH_DURATION_DEFAULT

        if json_data['radiant_win']:
            self.winner = RADIANT
        else:
            self.winner = DIRE

        self.heroes = [HeroDetails(hero_data) for hero_data in json_data['heroes']]

    def encode(self, output_array):
        if output_array.shape != [MatchEncodeMap.Total]:
            raise RuntimeError('Invalid shape')

        output_array[MatchEncodeMap.MatchDuration] = self.duration

        for i, hero in enumerate(self.heroes):
            begin_i = MatchEncodeMap.HeroStart + HeroEncodeMap.Total * i
            end_i = begin_i + HeroEncodeMap.Total
            hero.encode(output_array[begin_i:end_i])


if __name__ == '__main__':
    pass
