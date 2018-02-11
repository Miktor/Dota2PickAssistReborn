import os
import datetime
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from model.wrapper import ModelWrapper
from dota.json import *

PACKED_FILE = 'data/packed.json'
BATCH_SIZE = 64


def test_prediction(sess, model, picks_test, results_test):
    model.update_metrics(sess, picks_test, results_test)
    print(model.calc_metrics(sess))


def main():
    matches = matches_from_json_file(PACKED_FILE)
    model = ModelWrapper()

    picks_raw, results_raw = to_training_data(read())
    x_train, x_test, y_train, y_test = train_test_split(picks_raw, results_raw, train_size=0.8, random_state=13)

    timestamp = datetime.datetime.now().strftime('%d-%m-%Y %H-%M')
    train_writer = tf.summary.FileWriter(
        os.path.join('C:\\Development\\logs', timestamp), graph=tf.get_default_graph())


    epoch = 0
    while True:
        indices = np.random.randint(0, len(x_train), BATCH_SIZE)
        x_epoch = x_train[indices]
        y_epoch = y_train[indices]
        loss, acc, summ = model.train_picks(sess, 0.5, x_epoch, y_epoch)
        # train_writer.add_summary(summ, epoch)
        # print(pred_log)
        # print(pred)

        train_writer.add_summary(summ, epoch)

        if epoch % 100 == 0:
            print('{0} Loss: {1}; Acc: {2}'.format(epoch, loss, acc))
            hist = model.get_histograms(sess)
            train_writer.add_summary(hist, epoch)

        if loss < 0.005:
            print('Loss is {0} @ {1}, finished training'.format(loss, epoch))
            model.save()
            break

        epoch += 1

    print(model.evaluate(sess, x_test, y_test))
    # test_prediction(sess, model, x_test, y_test)


if __name__ == '__main__':
    main()
