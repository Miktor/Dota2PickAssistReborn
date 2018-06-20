from keras.layers import Dense, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend
from keras.callbacks import EarlyStopping, TerminateOnNaN

BATCH_SIZE = 256


class KerasModel(object):

    def __init__(self, sess, pick_shape: tuple, pick_game_shape: tuple, pick_game_num_actions: int):

        backend.set_session(sess)

        radiant_heroes = Input(shape=(pick_shape[0],), name='radiant_heroes')
        dire_heroes = Input(shape=(pick_shape[0],), name='dire_heroes')

        r1 = Dense(256, activation='relu')(radiant_heroes)
        d1 = Dense(256, activation='relu')(dire_heroes)

        rd = concatenate([r1, d1])
        x = Dense(512, activation='relu')(rd)
        x = Dense(512, activation='relu')(x)

        output = Dense(2, activation='softmax', name='output')(x)

        self.model = Model(inputs=[radiant_heroes, dire_heroes], outputs=[output])
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

    def train(self, picks, results, validation_picks, validation_results, max_epochs=1000):
        callbacks = [EarlyStopping(monitor='val_loss', patience=20), TerminateOnNaN()]
        self.model.fit(
                {'radiant_heroes': picks[0], 'dire_heroes': picks[1]},
                results,
                batch_size=BATCH_SIZE,
                epochs=max_epochs,
                callbacks=callbacks,
                validation_data=(
                    {'radiant_heroes': validation_picks[0], 'dire_heroes': validation_picks[1]},
                    validation_results))


    def predict(self, inputs):
        return self.model.predict(inputs)

    def evaluate(self, inputs, true_results, batch_size=128):
        return self.model.evaluate(inputs, true_results, batch_size=batch_size)


if __name__ == '__main__':
    pass
