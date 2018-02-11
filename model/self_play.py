from typing import List, Tuple
import numpy as np
import copy

import dota.common as common
from model.wrapper import ModelWrapper
from utils.mcts import MCTSNode, MCTSEnvModel, MCTSEstimator
from utils.utils import GenericMemory


def action_probabilities_to_policy(probabilities: np.ndarray, legal_actions: List[common.Action]) -> np.ndarray:
    action_indices = [common.ACTION_SPACE.index(action) for action in legal_actions]
    policy = np.zeros(len(common.ACTION_SPACE), dtype=np.float32)
    policy[action_indices] = probabilities
    return policy


class PickGameModel(MCTSEnvModel):
    def get_actions_for_state(self, state: common.PickGame):
        return state.get_legal_actions()

    def get_state_for_action(self, state: common.PickGame, action: common.Action):
        state_copy = copy.deepcopy(state)
        state_copy.do_action(action)
        return state_copy


class CmPickGameEstimator(MCTSEstimator):
    def __init__(self, model: ModelWrapper):
        self.model = model

    def predict(self, state: common.CaptainsModePickGame, actions: List[common.Action]) -> Tuple[np.ndarray, float]:
        policy, value = self.model.predict_policy_value(state)

        action_indices = [common.ACTION_SPACE.index(action) for action in actions]
        policy_for_actions = policy[action_indices]
        policy_for_actions = policy_for_actions / np.sum(policy_for_actions)  # Normalize across legal actions
        return policy_for_actions, value


class Simulation(object):
    def __init__(self, memory_size=10000, mcst_per_turn_simulations=100):
        self.model = ModelWrapper()
        self.mcts_estimator = CmPickGameEstimator(self.model)
        self.mcts_env_model = PickGameModel()

        self.memory = GenericMemory(memory_size, [
            ('state', np.float32, ()),
            ('policy', np.float32, ()),
            ('value', np.float32, ()),
        ])
        self.mcst_per_turn_simulations = mcst_per_turn_simulations

    def run(self, num_games=100):
        for game_i in range(num_games):
            s, p, r = self.play_game()
            print(game_i)

    def play_game(self):
        node = MCTSNode(common.CaptainsModePickGame())

        states = []
        policies = []

        while True:
            # Run MCST simulations
            node.run(self.mcts_env_model, self.mcts_estimator, self.mcst_per_turn_simulations)

            # Get move and selected node from MCTS
            move, new_node, probabilities = node.choose_action()

            # S, Policy,
            policies.append(action_probabilities_to_policy(probabilities, node.actions))
            states.append(node.state)

            if new_node.state.is_completed():
                radiant_prob, dire_prob = self.model.predict_pick_win(new_node.state.pick)

                outcomes = np.full(len(policies), dire_prob)
                # Outcomes for Radiant, since radiant moves first
                outcomes[::2] = radiant_prob
                return states, policies, outcomes
            else:
                node = new_node


if __name__ == '__main__':
    simulation = Simulation()
    simulation.run()
