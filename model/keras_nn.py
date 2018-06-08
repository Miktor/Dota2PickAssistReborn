import keras
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

BATCH_SIZE = 32


class KerasModel(object):

    def __init__(self, pick_shape: tuple, pick_game_shape: tuple, pick_game_num_actions: int):

        model = keras.Sequential()
        model.add(Dense(512, activation='relu', input_shape=(pick_shape[0],)))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
        self.model = model

    def train(self, picks, results, max_epochs=1000):
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=20),
            keras.callbacks.TerminateOnNaN()
        ]
        self.model.fit(picks, results,
                       batch_size=BATCH_SIZE,
                       epochs=max_epochs,
                       validation_split=0.2,
                       callbacks=callbacks)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def evaluate(self, inputs, true_results, batch_size=128):
        return self.model.evaluate(inputs, true_results, batch_size=batch_size)


if __name__ == '__main__':
    pass
