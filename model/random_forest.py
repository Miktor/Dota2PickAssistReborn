from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from xgboost import plot_tree
import matplotlib.pyplot as plt

LEARNING_RATE = 5 * 1e-5
L2_BETA = 0.001


class RFModel(object):

    def __init__(self, pick_shape: tuple, pick_game_shape: tuple, pick_game_num_actions: int):

        self.model = XGBClassifier(n_estimators=1000, n_jobs=4, max_depth=5, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic')

    def train_picks(self, inputs, results, inputs_test, results_test):
        self.model.fit(X=inputs, y=results,
                eval_set=[(inputs_test, results_test)],
                eval_metric='logloss;auc',
                verbose=True)
        return

    def predict_win(self, inputs):
        self.model.predict(inputs)
        plot_tree(self.model)
        plt.show()
        return

    def predict_policy_value(self, states):
        pass

    def evaluate(self, inputs, target_results):
        predictions = self.predict_win(inputs)
        return accuracy_score(target_results, predictions)


if __name__ == '__main__':
    pass