import os
import datetime
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split

from model.wrapper import ModelWrapper
from model.rf_wrapper import RFWrapper
from model.keras_wrapper import KerasModelWrapper
from dota.json import matches_from_json_file, matches_from_csv_file

PACKED_FILE = 'data/packed.json'
CSV_FILE = 'data/dota_dataset.csv'
BATCH_SIZE = 64


def test_prediction(sess, model, picks_test, results_test):
    model.update_metrics(sess, picks_test, results_test)
    print(model.calc_metrics(sess))

def tf_main():
    matches = matches_from_json_file(PACKED_FILE)
    model = ModelWrapper()

    timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M')
    train_writer = tf.summary.FileWriter(os.path.join('C:\\Development\\logs', timestamp), graph=tf.get_default_graph())

    matches_train, matches_test = train_test_split(matches, train_size=0.1, random_state=13)

    epoch = 0
    total_match_count = len(matches_train)
    while True:
        indices = np.random.randint(0, total_match_count, BATCH_SIZE)
        epoch_matches = [matches_train[i] for i in indices]
        loss, acc, summ = model.train_picks(epoch_matches)
        # train_writer.add_summary(summ, epoch)
        # print(pred_log)
        # print(pred)

        train_writer.add_summary(summ, epoch)

        if epoch % 100 == 0:
            print('{0} Loss: {1}; Acc: {2}'.format(epoch, loss, acc))
            # hist = model.get_histograms(sess)
            # train_writer.add_summary(hist, epoch)

        if loss < 0.005:
            print('Loss is {0} @ {1}, finished training'.format(loss, epoch))
            # model.save()
            # break

        epoch += 1

    # print(model.evaluate(sess, x_test, y_test))
    # test_prediction

def run_model(model):
    matches = matches_from_json_file(PACKED_FILE)

    matches_train, matches_test = train_test_split(matches, train_size=0.8, random_state=13)

    model.train_picks(matches_train, matches_test)

def get_data(dataset):
    #radiant_data, dire_data = dataset.ix[:,:108], dataset.ix[:,108:216]
    data = dataset.ix[:,:216]
    winners = pd.get_dummies(dataset['radiant_win'])
    #return [radiant_data, dire_data], winners
    return data, winners

def run_csv_model(model):
    dataset = matches_from_csv_file(CSV_FILE)

    train, test = train_test_split(dataset, test_size=0.2, random_state=13)
    a, b = get_data(train)
    c, d = get_data(test)
    model.train_picks_raw(a, b, c, d)

def rf_main():
    run_csv_model(RFWrapper())


def keras_main():
    run_csv_model(KerasModelWrapper())


if __name__ == '__main__':
    rf_main()
    # keras_main()
