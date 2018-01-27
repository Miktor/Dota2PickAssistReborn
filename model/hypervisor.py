from enum import IntEnum, auto

import numpy as np

from model.heroes import Hero, Lane, Role
from model.input_data import HeroEncodeMap, MatchEncodeMap
from model.mcts import MCTSNode
from model.pick_prediction_model import PickPredictionModel
from model.picker_model import GraphPredictionModel
from model.heroes import NUM_HEROES



class GamePhases(IntEnum):
    SelectHero = auto()
    SelectHeroRoleAndLane = auto()
    Ban = auto()
    Play = auto()



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

    def __init__(self, game_mode: GameMode, phase: GamePhases, picked_heroes=[], banned_heroes=[]):
        self.current_phase = phase
        self.picked_heroes = picked_heroes
        self.banned_heroes = banned_heroes

        self.game_mode = game_mode

    def get_next_state(self, action: Action):
        return self.game_mode.next(self, action)



class AllPickMode(GameMode):

    def __init__(self):
        super(AllPickMode, self).__init__(GamePhases.SelectHero)

    def next(self, current_state: GameState, action: Action):
        if len(current_state.picked_heroes) == 5:
            return GamePhases.Start
        else:
            # create new instance of state values
            picked_heroes = list(current_state.picked_heroes)
            banned_heroes = list(current_state.banned_heroes)
            if type(action) is SelectHero:
                picked_heroes.append(action.hero)
                banned_heroes.append(action.hero)

            return GameState(self, GamePhases.SelectHero, picked_heroes, banned_heroes)

    def first_state(self):
        return GameState(self, GamePhases.SelectHero)



class GameModel(object):

    def get_state_for_action(self, state: GameState, action: Action):
        return state.get_next_state(action)

    def get_actions_for_state(self, state: GameState):
        if state.current_phase == GamePhases.SelectHero:
            if len(state.picked_heroes) == 5:
                return
            return list(map(SelectHero, Hero))
        pass



class GameResultEstimator(object):

    def __init__(self):
        self.model = GraphPredictionModel(1, 1)

    @staticmethod
    def build_data(state: GameState, actions):
        initial_state_data = np.zeros(shape=[HeroEncodeMap.Total * 5 + ActionMap.Total], dtype=np.float32)
        for i, hero in enumerate(state.picked_heroes):
            begin_i = MatchEncodeMap.HeroStart + HeroEncodeMap.Total * i
            end_i = begin_i + HeroEncodeMap.Total
            hero.encode(initial_state_data[begin_i:end_i])

        output_data = np.tile(initial_state_data, [len(actions), 1])

        for i, action in enumerate(actions):
            if type(action) is SelectHero:
                action_data = output_data[i, HeroEncodeMap.Total * 5:]
                action.encode(action_data)

        return output_data

    def predict(self, state: GameState, actions):
        data_to_predict = GameResultEstimator.build_data(state, actions)

        return np.full((len(actions)), 0.5), 0.5



def play():
    estimator = GameResultEstimator()

    game_model = GameModel()
    game_mode = AllPickMode()
    state = game_mode.first_state()

    mcst = MCTSNode(state)

    for x in range(1, 6):
        mcst.run(game_model, estimator, 100)
        action, child, _ = mcst.choose_action()
        print('Pick #{}: {}'.format(x, str(action.hero.hero)))
        mcst = child

    print('done')



if __name__ == '__main__':
    play()
