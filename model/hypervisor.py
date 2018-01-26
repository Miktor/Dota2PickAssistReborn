from enum import IntEnum, auto

from model.mcts import MCTSNode
from model.pick_prediction_model import PickPredictionModel
from model.picker_model import GraphPredictionModel
from model.heroes import encode_hero, load_heroes, Hero, Lane, Role
from model.input_data import InputData, MatchEncodeMap, ResultsEncodeMap


class GamePhases(IntEnum):
    SelectHero = auto()
    SelectHeroRoleAndLane = auto()
    Ban = auto()
    Play = auto()


class PickedHero:

    def __init__(self, hero: Hero, lane=Lane.Mid, role=Role.Carry):
        self.hero = hero  # Hero
        self.lane = lane  # Lane
        self.role = role  # Role


class GameState(object):

    def __init__(self, phase: GamePhases):
        self.current_phase = phase
        self.picked_heroes = []

    def get_next_state(self):
        if len(self.picked_heroes) == 5:
            return GamePhases.Start
        else:
            return game_mode.next()


class GameMode(object):

    def __init__(self, phase: GamePhases):
        self.current_phase = phase


class AllPickMode(GameMode):

    def __init__(self):
        super(AllPickMode, self).__init__(GamePhases.SelectHero)

    def next(self):
        return GamePhases.SelectHero

    def first_state(self):
        return GameState(GamePhases.SelectHero)


class Action(object):
    pass


class SelectHero(Action):

    def __init__(self, hero: Hero):
        self.hero = hero


class GameModel(object):

    def get_state_for_action(self, actions):
        pass

    def get_actions_for_state(self, state: GameState):
        if state.current_phase == GamePhases.SelectHero:
            return list(map(SelectHero, Hero))
        pass


class GameResultEstimator(object):

    def __init__(self):
        self.model = PickPredictionModel(1, 1)
        
    def _bild_game_state(self, state: GameState):
        pass

    def _bild_data(self, state: GameState):
        pass

    def predict(state: GameState, actions:Action[]):
        converted_state = _bild_game_state(state)
        data_to_predict = _bild_data(converted_state, actions)


def play():

    estimator = PickPredictionModel(1, 1)

    game_model = GameModel()
    game_mode = AllPickMode()
    state = game_mode.first_state()

    mcst = MCTSNode(state)
    mcst.run(game_model, estimator, 100)


if __name__ == '__main__':
    play()
