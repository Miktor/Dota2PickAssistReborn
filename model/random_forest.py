import xgboost as xgb
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from xgboost import plot_tree
import matplotlib.pyplot as plt

LEARNING_RATE = 5 * 1e-5
L2_BETA = 0.001


class RFModel(object):

    def __init__(self, pick_shape: tuple, pick_game_shape: tuple, pick_game_num_actions: int):

        # set xgboost params
        self._param = {
            'booster': 'dart',
            'max_depth': 5,  # the maximum depth of each tree
            'eta':       0.1,  # the training step for each iteration
            'learning_rate': 0.1,
            'sample_type': 'uniform',
            'normalize_type': 'tree',
            'rate_drop': 0.1,
            'skip_drop': 0.5,
            'gamma':     0.5,  # minimum loss reduction required to make a further partition
            'min_child_weight': 1,
            'subsample': 0.8,
            'silent':    1,  # logging mode - quiet
            'nthread':   4,
            'objective': 'binary:logistic',  # error evaluation for multiclass training
            'eval_metric': "auc",
            'num_class': 1}  # the number of classes that exist in this datset


    def train_picks(self, inputs, results, inputs_test, results_test):
        # use DMatrix for xgbosot
        train = xgb.DMatrix(inputs, label=results)
        test = xgb.DMatrix(inputs_test, label=results_test)

        num_round = 10  # the number of training iterations

        bst = xgb.train(self._param, train, num_round)

        preds = bst.predict(test)

        accuracy = accuracy_score(results_test, preds.round())
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        prec = average_precision_score(results_test, preds)
        print("Precision: %.2f%%" % (prec * 100.0))
        roc_auc_score(results_test, preds)
        print("ROC AUC: %.2f%%" % (prec * 100.0))

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