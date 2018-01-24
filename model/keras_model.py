from keras import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

class KerasModel(object):

    def __init__(self, input_shape, outputs):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_shape=input_shape))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(outputs, activation='softmax'))

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        self.model = model

    def train(self, inputs, target_results, batch_size=128, epochs=10):
        self.model.fit(inputs, target_results, batch_size=batch_size, epochs=epochs)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def evaluate(self, inputs, true_results, batch_size=128):
        return self.model.evaluate(inputs, true_results, batch_size=batch_size)
