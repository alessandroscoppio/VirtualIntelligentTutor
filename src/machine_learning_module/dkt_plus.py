from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Masking, TimeDistributed
from keras.layers import Flatten
from keras.models import load_model


class LSTMModel:
    def __init__(self, hidden_units, n_exercises, batch_size):
        self.n_exercises = n_exercises
        self.batch_size = batch_size

        # define model
        self.history = None
        self.model = Sequential()

        # ignore timesteps containing -1s dealing with padding
        self.model.add(Masking(-1., batch_input_shape = (batch_size, None, 2 * n_exercises)))
        self.model.add(
            LSTM(units = hidden_units, input_dim = n_exercises * 2, return_sequences = True, stateful = True))
        self.model.add(Dropout(0.5))
        self.model.add(TimeDistributed(Dense(n_exercises, activation = 'sigmoid')))
        self.model.add(Dense(self.n_exercises + 1, activation = 'sigmoid'))
        self.model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

    # def fit(self, X, y, epochs, verbose=0, batch_size=32):
    def fit(self, train_gen, val_gen, epochs, verbose = 0, batch_size = 32):
        # fit model
        # self.history = self.model.fit(X, y, epochs=epochs, verbose=verbose, batch_size=batch_size)

        # try to use fit generator to try their dataset with less efforts
        self.history = self.model.fit_generator(shuffle = False,
                                                validation_data = val_gen.get_generator(),
                                                validation_steps = val_gen.total_steps,
                                                epochs = epochs,
                                                steps_per_epoch = train_gen.total_steps,
                                                generator = train_gen.get_generator(),
                                                callbacks = None,
                                                verbose = verbose)

    def predict(self, input):
        # demonstrate prediction
        x_input = input.reshape((1, self.max_n_interactions, 1))
        prediction = self.model.predict(x_input, verbose = 0)
        return prediction

    def save_model(self, name):
        self.model.save('saved-models/' + name)

    def load_model(self, name):
        self.model = load_model(name)
