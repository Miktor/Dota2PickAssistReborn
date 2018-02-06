from dota.heroes import get_hero_id, NUM_HEROES, Lane, Role, Hero
import enum

RADIANT = 1.0
DIRE = -1.0
MATCH_DURATION_DEFAULT = 60 * 45.0  # 45 min as 1.0


class HeroEncodeMap(enum.IntEnum):
    HeroId = 0
    HeroLane = NUM_HEROES
    HeroRole = HeroLane + Lane.Total
    Total = HeroRole + Role.Total


class MatchEncodeMap(enum.IntEnum):
    MatchDuration = 0
    HeroStart = 1
    HeroEnd = HeroStart + HeroEncodeMap.Total * 10
    Total = HeroEnd


class ResultsEncodeMap(enum.IntEnum):
    Radiant = 0
    Dire = 1
    Total = 2


PARAMETERS_PER_MATCH = 1
PARAMETERS_PER_HERO = NUM_HEROES



def encode_hero(output_data, hero_index, hero_id: Hero, side, lane, lane_role, is_roaming):
    begin_i = MatchEncodeMap.HeroStart + HeroEncodeMap.Total * hero_index
    end_i = begin_i + HeroEncodeMap.Total

    hero_data = output_data[begin_i:end_i]

    hero_data[hero_id] = side

    hero_data[HeroEncodeMap.HeroLane + lane - 1] = 1
    hero_data[HeroEncodeMap.HeroRole + lane_role - 1] = 1


def encode_match(output_data, duration):
    if output_data.shape[0] != MatchEncodeMap.Total:
        raise RuntimeError('Invalid shape')

    output_data[MatchEncodeMap.MatchDuration] = duration


def encode_hero_from_json(output_data, hero_index, hero_data):

    hero_id = get_hero_id(hero_id=hero_data['hero_id'])

    if hero_data['isRadiant']:
        side = RADIANT
    else:
        side = DIRE

    lane = hero_data['lane']
    lane_role = hero_data['lane_role']
    is_roaming = bool(hero_data['is_roaming'])

    encode_hero(output_data, hero_index, hero_id, side, lane, lane_role, is_roaming)


def encode_from_json(json_data, output_data, output_results):
    duration = json_data['duration'] / MATCH_DURATION_DEFAULT

    encode_match(output_data, duration)

    radiant_win = bool(json_data['radiant_win'])
    if radiant_win:
        output_results[ResultsEncodeMap.Radiant] = 1
    else:
        output_results[ResultsEncodeMap.Dire] = 1

    for i, hero_data in enumerate(json_data['heroes']):
        encode_hero_from_json(output_data, i, hero_data)


if __name__ == '__main__':
    pass
