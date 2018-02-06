from model.mcts import MCTSNode
from utils.utils import GenericMemory
from dota.dota_pick_model import *
from dota.input_data import *
from model.pick_prediction_model import PickPredictionModel
import tensorflow as tf


class MatchResultPredictor(object):
    def __init__(self):
        self.sess = None
        self.model = PickPredictionModel(picks_inputs=MatchEncodeMap.Total, picks_outputs=ResultsEncodeMap.Total)

    def predict_match_result(self, radiant_pick, dire_pick):
        predict_data = np.zeros(shape=[1, MatchEncodeMap.Total], dtype=np.float32)
        encode_match(predict_data[0, :], MATCH_DURATION_DEFAULT)

        for i, hero in enumerate(radiant_pick):
            encode_hero(predict_data[0, :], i, hero.hero, RADIANT, hero.lane, hero.role, False)
        for i, hero in enumerate(dire_pick):
            encode_hero(predict_data[0, :], i + 5, hero.hero, DIRE, hero.lane, hero.role, False)

        return self.model.predict_win(self.sess, predict_data)

    def predict(self, state, actions):
        return self.model.predict_picks(self.sess, actions)


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
        match_result_predictor = MatchResultPredictor()

        game_model = GameModel()
        game_mode = AllPickMode()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.4

        with tf.Session(config=config) as sess:
            match_result_predictor.sess = sess
            match_result_predictor.model.load_if_exists(sess)

            for game_i in range(num_games):
                s, p, r = self.play_game(sess, match_result_predictor, game_model, game_mode)
                print(game_i)

    def play_game(self, sess: tf.Session, match_result_predictor, game_model, game_mode):
        node = MCTSNode(game_mode.first_state())

        states = []
        policies = []

        while True:
            # Run MCST simulations
            node.run(game_model, match_result_predictor, self.mcst_per_turn_simulations)

            # Get move and selected node from MCTS
            move, new_node, probabilities = node.choose_action()

            # S, Policy,
            policies.append(action_probabilities_to_policy(probabilities, node.actions))
            states.append(node.state)

            if new_node.state.is_finished():
                result = match_result_predictor.predict_match_result(sess, new_node.state.radiant_heroes,
                                                                     new_node.state.dire_heroes)
                radiant_value = result[0]
                dire_value = result[1]

                outcomes = np.full(len(policies), dire_value)
                # Outcomes for Radiant, since radiant moves first
                outcomes[::2] = radiant_value
                return states, policies, outcomes
            else:
                assert len(new_node.state.radiant_heroes) > len(node.state.radiant_heroes) or len(new_node.state.dire_heroes) > len(node.state.dire_heroes)
                node = new_node



if __name__ == '__main__':

    simulation = Simulation()
    simulation.run()
