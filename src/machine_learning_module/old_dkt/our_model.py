import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Masking, TimeDistributed
from keras.layers import Flatten
from keras.models import load_model
from keras.utils import Progbar
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
from keras import backend as K

from Utils import DataGenerator

'''JUST TO TRY IF IT WORKS, REFACTOR'''


# This method is for internal use. You should not use it outside of this file.
def model_evaluate(test_gen, model, metrics, verbose=0):
    def predict():
        def get_target_skills(preds, labels):
            target_skills = labels[:, :, 0:test_gen.num_skills]
            target_labels = labels[:, :, test_gen.num_skills]

            target_preds = np.sum(preds * target_skills, axis=2)

            return target_preds, target_labels

        y_true_t = []
        y_pred_t = []
        test_gen.reset()

        while not test_gen.done:
            # Get batch
            batch_features, batch_labels = test_gen.next_batch()

            # Predict
            predictions = model.predict_on_batch(batch_features)

            # Get target skills
            target_preds, target_labels = get_target_skills(predictions, batch_labels)
            flat_pred = np.reshape(target_preds, [-1])
            flat_true = np.reshape(target_labels, [-1])

            # Remove mask
            mask_idx = np.where(flat_true == -1.0)[0]
            flat_pred = np.delete(flat_pred, mask_idx)
            flat_true = np.delete(flat_true, mask_idx)

            # Save it
            y_true_t.extend(flat_true)
            y_pred_t.extend(flat_pred)

            if verbose and test_gen.step < test_gen.total_steps:
                progbar.update(test_gen.step)

        return y_true_t, y_pred_t

    # assert (isinstance(test_gen, DataGenerator))
    assert (model is not None)
    assert (metrics is not None)

    if verbose:
        print("==== Evaluation Started ====")

    progbar = Progbar(target=test_gen.total_steps, verbose=verbose)

    y_true, y_pred = predict()

    bin_pred = [1 if p > 0.5 else 0 for p in y_pred]

    results = {}
    if 'auc' in metrics:
        results['auc'] = roc_auc_score(y_true, y_pred)
    if 'acc' in metrics:
        results['acc'] = accuracy_score(y_true, bin_pred)
    if 'pre' in metrics:
        results['pre'] = precision_score(y_true, bin_pred)

    if verbose:
        progbar.update(test_gen.step, results.items())
        print("==== Evaluation Done ====")

    return results


class LSTMModel:
    def __init__(self, hidden_units, n_exercises, batch_size):
        self.n_exercises = n_exercises
        self.batch_size = batch_size

        # define custom crossentropy loss since keras add one value
        def loss_function(real_label, prediction):
            target_skills = real_label[:, :, 0:n_exercises]
            target_labels = real_label[:, :, n_exercises]
            target_preds = K.sum(prediction * target_skills, axis=2)

            return K.binary_crossentropy(target_labels, target_preds)
        # define model
        self.history = None
        self.model = Sequential()

        # ignore timesteps containing -1s dealing with padding
        self.model.add(Masking(-1., batch_input_shape=(batch_size, None, 2 * n_exercises)))
        self.model.add(LSTM(units=hidden_units, return_sequences=True, stateful=True))
        self.model.add(Dropout(0.5))
        self.model.add(TimeDistributed(Dense(n_exercises, activation='sigmoid')))
        # self.model.add(Dense(self.n_exercises+1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss=loss_function)
        self.model.summary()

    def fit(self, train_gen, val_gen, epochs, verbose=2, batch_size=32):
        # fit model
        # self.history = self.model.fit(X, y, epochs=epochs, verbose=verbose, batch_size=batch_size)

        # try to use fit generator to try their dataset with less efforts
        log_dir = "logs/models"
        checkpoint_filename = os.path.join(log_dir, "weights.model")
        model_checkpoint_callback = ModelCheckpoint(checkpoint_filename, save_best_only=True, verbose=1, monitor="val_loss", mode='min')
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode="min")

        self.history = self.model.fit_generator(shuffle=False,
                                                validation_data=val_gen.get_generator(),
                                                validation_steps=val_gen.total_steps,
                                                epochs=epochs,
                                                steps_per_epoch=train_gen.total_steps,
                                                generator=train_gen.get_generator(),
                                                callbacks=[model_checkpoint_callback, early_stopping_callback],
                                                verbose=verbose)

    def predict_from_student(self, student_submissions):

        student_submissions = np.array(student_submissions).reshape((1, len(student_submissions), 2*self.n_exercises))
        prediction = self.model.predict(student_submissions, verbose=0)
        return prediction

    def evaluate(self, test_gen, metrics, verbose=2, filepath_log=None):
        results = model_evaluate(test_gen, self.model, metrics, verbose)

        if filepath_log is not None:
            with open(filepath_log, 'w') as fl:
                fl.write("auc,acc,pre\n{0},{1},{2}".format(results['auc'], results['acc'], results['pre']))

        return results

    def save_model(self, name):
        self.model.save('saved_models/' + name)

    def load_model(self, name):
        self.model = load_model(name)