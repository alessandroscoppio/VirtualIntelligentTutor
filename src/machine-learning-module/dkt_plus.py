from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.layers import Flatten
from keras.models import load_model


class LSTMModel:
    def __init__(self, n_interactions, n_exercises):
        self.n_interactions = n_interactions
        self.n_exercises = n_exercises
        # define model
        self.history = None
        self.model = Sequential()
        self.model.add(LSTM(200, activation='relu', input_shape=(self.n_interactions, n_exercises)))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.n_exercises))
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def fit(self, X, y, epochs, verbose=0, batch_size=32):
        # fit model
        self.history = self.model.fit(X, y, epochs=epochs, verbose=verbose, batch_size=batch_size)

    def predict(self, input):
        # demonstrate prediction
        x_input = input.reshape((1, self.input_size, 1))
        prediction = self.model.predict(x_input, verbose=0)
        return prediction

    def save_model(self, name):
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)
