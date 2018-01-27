import numpy as np

from model.mcts import MCTSNode
from utils.utils import GenericMemory
from model.dota_pick_model import *


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



class MatchResultPredictor(object):

    def predict_match_result(self, radiant_pick, dire_pick):
        return [0.5, 0.5]



def action_probabilities_to_policy(probabilities: np.ndarray, legal_actions) -> np.ndarray:
    return [0, 0, 0, .2, 0.5]



class Simulation(object):

    def __init__(self, memory_size=10000, mcst_per_turn_simulations=100):
        self.memory = GenericMemory(memory_size, [
            ('state', np.float32, ()),
            ('policy', np.float32, ()),
            ('value', np.float32, ()),
        ])
        self.mcst_per_turn_simulations = mcst_per_turn_simulations

    def run(self, num_games=100):
        estimator = GameResultEstimator()
        game_model = GameModel()
        game_mode = AllPickMode()

        for game_i in range(num_games):
            s, p, r = self.play_game(estimator, game_model, game_mode)
            print(game_i)

    def play_game(self, estimator, game_model, game_mode):
        node = MCTSNode(game_mode.first_state())
        match_result_predictor = MatchResultPredictor()

        states = []
        policies = []

        while True:
            # Run MCST simulations
            node.run(game_model, estimator, self.mcst_per_turn_simulations)

            # Get move and selected node from MCTS
            move, new_node, probabilities = node.choose_action()

            # S, Policy,
            policies.append(action_probabilities_to_policy(probabilities, node.actions))
            states.append(node.state)

            if new_node.state.is_finished():
                result = match_result_predictor.predict_match_result(new_node.state.picked_heroes,
                                                                     new_node.state.picked_heroes)
                radiant_value = result[0]
                dire_value = result[1]

                outcomes = np.full(len(policies), dire_value)
                # Outcomes for Radiant, since radiant moves first
                outcomes[::2] = radiant_value
                return states, policies, outcomes
            else:
                assert len(new_node.state.picked_heroes) > len(node.state.picked_heroes)
                node = new_node



if __name__ == '__main__':
    simulation = Simulation()
    simulation.run(100)
