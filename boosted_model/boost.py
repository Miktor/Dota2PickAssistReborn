import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import accuracy_score


PACKED_FILE = 'data\\learn.txt'

X, y = datasets.load_svmlight_file(PACKED_FILE, dtype=np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# use DMatrix for xgbosot
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# use svmlight file for xgboost
dump_svmlight_file(X_train, y_train, 'boosted_model/dtrain.svm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'boosted_model/dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('boosted_model/dtrain.svm')
dtest_svm = xgb.DMatrix('boosted_model/dtest.svm')

# set xgboost params
param = {
    'max_depth': 5,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'binary:logistic',  # error evaluation for multiclass training
    'num_class': 1}  # the number of classes that exist in this datset
num_round = 20  # the number of training iterations

#------------- numpy array ------------------
# training and testing - numpy matrices
bst = xgb.train(param, dtrain, num_round)
preds = bst.predict(dtest)

accuracy = accuracy_score(y_test, preds)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# # extracting most confident predictions
# best_preds = np.asarray([np.argmax(line) for line in preds])
# print("Numpy array precision:", precision_score(y_test, best_preds, average='varloss'))
#
# # ------------- svm file ---------------------
# # training and testing - svm file
# bst_svm = xgb.train(param, dtrain_svm, num_round)
# preds = bst.predict(dtest_svm)
#
# # extracting most confident predictions
# best_preds_svm = [np.argmax(line) for line in preds]
# print("Svm file precision:",precision_score(y_test, best_preds_svm, average='varloss'))
# # --------------------------------------------
#
# # dump the models
# bst.dump_model('dump.raw.txt')
# bst_svm.dump_model('dump_svm.raw.txt')
#
#
# # save the models for later
# joblib.dump(bst, 'bst_model.pkl', compress=True)
# joblib.dump(bst_svm, 'bst_svm_model.pkl', compress=True)