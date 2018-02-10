import numpy as np

from enum import IntEnum
from dota.heroes import Hero, Lane, NUM_HEROES, Role
from dota.input_data import HeroEncodeMap, MatchEncodeMap



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

    @staticmethod
    def get_hero_index(hero: Hero):
        return list(Hero).index(hero)

    def encode(self, output_data):
        output_data[PickedHero.get_hero_index(self.hero)] = 1

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
        return list(map(SelectHero, Hero))


class GameResultEstimator(object):
    def __init__(self):
        self.model = GraphPredictionModel(1, 1)

    @staticmethod
    def build_data(state: GameState, actions):

        initial_state_data = np.zeros(shape=[HeroEncodeMap.Total * 5 + ActionMap.Total], dtype=np.float32)
        for i, hero in enumerate(state.radiant_heroes):
            begin_i = MatchEncodeMap.HeroStart + HeroEncodeMap.Total * i
            end_i = begin_i + HeroEncodeMap.Total
            hero.encode(initial_state_data[begin_i:end_i])

        for i, hero in enumerate(state.dire_heroes):
            begin_i = MatchEncodeMap.HeroStart + HeroEncodeMap.Total * i
            end_i = begin_i + HeroEncodeMap.Total
            hero.encode(initial_state_data[begin_i:end_i])

        if actions is None:
            return np.tile(initial_state_data, 1)

        output_data = np.tile(initial_state_data, [len(actions), 1])

        for i, action in enumerate(actions):
            if type(action) is SelectHero:
                action_data = output_data[i, HeroEncodeMap.Total * 5:]
                action.encode(action_data)

        return output_data

    def predict(self, state: GameState, actions):
        data_to_predict = GameResultEstimator.build_data(state, actions)

        return np.full(data_to_predict.shape[0], 0.5), 0.5


if __name__ == '__main__':
    pass
