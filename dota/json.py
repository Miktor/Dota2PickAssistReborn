from typing import List
import json
import dota.common as common
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

def picked_hero_from_json(json_obj: dict) -> common.PickedHero:
    hero = common.Hero(json_obj['hero_id'])
    lane = common.Lane(json_obj['lane'])
    lane_role = common.Role(json_obj['lane_role'])
    is_roaming = bool(json_obj['is_roaming'])
    return common.PickedHero(hero, lane=lane, role=lane_role)


def pick_from_json(json_obj: dict) -> common.Pick:
    pick = common.Pick()

    for hero_data in json_obj['heroes']:
        if hero_data['isRadiant']:
            side = common.Side.RADIANT
        else:
            side = common.Side.DIRE
        picked_hero = picked_hero_from_json(hero_data)
        pick.append(picked_hero, side)
    return pick


def match_from_json(json_obj: dict) -> common.Match:
    duration = json_obj['duration'] / common.Match.DEFAULT_DURATION
    radiant_win = bool(json_obj['radiant_win'])
    side = common.Side.RADIANT if radiant_win else common.Side.DIRE
    pick = pick_from_json(json_obj)

    return common.Match(pick, side, duration)


def matches_from_json(data: list) -> List[common.Match]:
    matches = []
    for match_data in data:
        matches.append(match_from_json(match_data))
    return matches


def matches_from_json_file(path: str) -> List[common.Match]:
    with open(path, 'r') as f:
        data = json.loads(f.read())
        return matches_from_json(data)

def matches_from_csv_file(path: str):
    dataset = pd.read_csv(path, index_col=0)
    dataset = dataset.take(np.random.permutation(len(dataset)))

    x = dataset.drop('radiant_win', axis=1)
    y = dataset['radiant_win']

    print('Logistic Regression accuracy:', np.mean(cross_val_score(LogisticRegression(), x, y, scoring='accuracy', cv = 2)))
    print('MultinominalNB accuracy:', np.mean(cross_val_score(MultinomialNB(), x, y, scoring='accuracy', cv = 2)))
    return dataset
