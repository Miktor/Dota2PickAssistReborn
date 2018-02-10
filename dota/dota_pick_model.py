import numpy as np

from enum import IntEnum
from dota.heroes import Hero, Lane, NUM_HEROES, Role
from dota.input_data import HeroEncodeMap, MatchEncodeMap, encode_hero, RADIANT, DIRE, StateEncodeMap


def get_hero_index(hero: Hero):
    return list(Hero).index(hero)


class GamePhases(IntEnum):
    RadiantSelectHero = 0
    DireSelectHero = 1
    Ban = 2
    Play = 3

class ActionMap(IntEnum):
    HeroId = 0
    HeroLane = NUM_HEROES
    HeroRole = HeroLane + Lane.Total
    Total = HeroRole + Role.Total


class PickedHero:
    def __init__(self, hero: Hero, lane=Lane.NoLane, role=Role.NoRole):
        self.hero = hero  # Hero
        self.lane = lane  # Lane
        self.role = role  # Role

    def encode(self, output_data):
        output_data[get_hero_index(self.hero)] = 1

        if self.lane:
            output_data[HeroEncodeMap.HeroLane + self.lane - 1] = 1
        if self.role:
            output_data[HeroEncodeMap.HeroRole + self.lane_role - 1] = 1


class Action(object):
    pass


class SelectHero(Action):
    def __init__(self, hero: Hero):
        self.hero = PickedHero(hero)

    def encode(self, out_data):
        self.hero.encode(out_data)


class GameMode(object):
    def __init__(self, phase: GamePhases):
        self.current_phase = phase

    def next(self, state: 'GameState', action: Action):
        raise NotImplementedError


class GameState(object):
    def __init__(self, game_mode: GameMode, phase: GamePhases, radiant_heroes=[], dire_heroes=[], banned_heroes=[]):
        self.current_phase = phase  # type: GamePhases
        self.radiant_heroes = radiant_heroes
        self.dire_heroes = dire_heroes
        self.banned_heroes = banned_heroes

        self.game_mode = game_mode  # type: GameMode

    def get_next_state(self, action: Action):
        return self.game_mode.next(self, action)

    def is_finished(self):
        return len(self.radiant_heroes) == len(self.dire_heroes) == 5

    def encode(self):
        predict_data = np.zeros(StateEncodeMap.Total, dtype=np.float32)
        predict_data[StateEncodeMap.Phase] = self.current_phase
        for i, hero in enumerate(self.radiant_heroes):
            encode_hero(predict_data[StateEncodeMap.RadiantPick:], 0, i, hero.hero, RADIANT, 0, 0, False)
        for i, hero in enumerate(self.dire_heroes):
            encode_hero(predict_data[StateEncodeMap.DirePick:], 0, i + 5, hero.hero, DIRE, 0, 0, False)
        for i, hero in enumerate(self.banned_heroes):
            encode_hero(predict_data[StateEncodeMap.BannedHeroes:], 0, 0, hero.hero, 1., 0, 0, False)
        return predict_data


class AllPickMode(GameMode):
    def __init__(self):
        super(AllPickMode, self).__init__(GamePhases.RadiantSelectHero)

    def next(self, current_state: GameState, action: Action):
        if len(current_state.radiant_heroes) == 5 and len(current_state.dire_heroes) == 5:
            return GamePhases.Start
        else:
            banned_heroes = list(current_state.banned_heroes)
            radiant_heroes = current_state.radiant_heroes
            dire_heroes = current_state.dire_heroes

            # create new instance of state values
            new_phase = GamePhases.RadiantSelectHero
            if current_state.current_phase == GamePhases.RadiantSelectHero:
                radiant_heroes = list(current_state.radiant_heroes)
                radiant_heroes.append(action.hero)
                new_phase = GamePhases.DireSelectHero
            elif current_state.current_phase == GamePhases.DireSelectHero:
                dire_heroes = list(current_state.dire_heroes)
                dire_heroes.append(action.hero)
                new_phase = GamePhases.RadiantSelectHero
            else:
                assert (False)

            banned_heroes.append(action.hero)

            if type(action) is SelectHero:
                banned_heroes.append(action.hero)

            return GameState(self, new_phase, radiant_heroes, dire_heroes, banned_heroes)

    def first_state(self):
        return GameState(self, GamePhases.RadiantSelectHero)


class GameModel(object):
    def get_state_for_action(self, state: GameState, action: Action):
        return state.get_next_state(action)

    def get_actions_for_state(self, state: GameState):
        if state.is_finished():
            return

        return list(map(SelectHero, set(Hero) - set([v.hero for v in state.radiant_heroes] + [v.hero for v in state.dire_heroes] + state.banned_heroes)))


if __name__ == '__main__':
    pass
