from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

BATCH_SIZE = 32


class KerasModel(object):

    def __init__(self, pick_shape: tuple, pick_game_shape: tuple, pick_game_num_actions: int):

        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=(pick_shape[0],)))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        self.model = model

    def train(self, picks, results, epochs=10):
        self.model.fit(picks, results, batch_size=BATCH_SIZE, epochs=epochs)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def evaluate(self, inputs, true_results, batch_size=128):
        return self.model.evaluate(inputs, true_results, batch_size=batch_size)


if __name__ == '__main__':
    pass
