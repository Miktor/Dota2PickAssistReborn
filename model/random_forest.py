from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

LEARNING_RATE = 5 * 1e-5
L2_BETA = 0.001


class RFModel(object):

    def __init__(self, pick_shape: tuple, pick_game_shape: tuple, pick_game_num_actions: int):

        self.model = XGBClassifier()

    def train_picks(self, inputs, results, inputs_test, results_test):
        self.model.fit(X=inputs, y=results,
                eval_set=[(inputs_test, results_test)],
                eval_metric='logloss',
                verbose=True)
        return

    def predict_win(self, inputs):
        return self.model.predict(inputs)

    def predict_policy_value(self, states):
        pass

    def evaluate(self, inputs, target_results):
        predictions = self.predict_win(inputs)
        return accuracy_score(target_results, predictions)


if __name__ == '__main__':
    pass