import pandas as pd
import random
import math
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def split_tuple(dt):
    return [[value[0] for value in seq] for seq in dt], [[value[1] for value in seq] for seq in dt]


def split_dataset(data, validation_rate, testing_rate, shuffle=True):
    seqs = data
    if shuffle:
        random.shuffle(seqs)

    # Get testing data
    test_idx = random.sample(range(0, len(seqs) - 1), int(len(seqs) * testing_rate))
    X_test, y_test = split_tuple([value for idx, value in enumerate(seqs) if idx in test_idx])
    seqs = [value for idx, value in enumerate(seqs) if idx not in test_idx]

    # Get validation data
    val_idx = random.sample(range(0, len(seqs) - 1), int(len(seqs) * validation_rate))
    X_val, y_val = split_tuple([value for idx, value in enumerate(seqs) if idx in val_idx])

    # Get training data
    X_train, y_train = split_tuple([value for idx, value in enumerate(seqs) if idx not in val_idx])

    return X_train, X_val, X_test, y_train, y_val, y_test


def read_file(dataset_path):
    data = pd.read_csv(dataset_path, dtype={'skill_name': str})

    # Step 1 - Remove problems without a skill_id
    data.dropna(subset=['skill_id'], inplace=True)

    # Step 2 - Convert to sequence by student id
    students_seq = data.groupby("user_id", as_index=True)["skill_id", "correct"].apply(
        lambda x: x.values.tolist()).tolist()

    # Step 3 - Rearrange the skill_id
    seqs_by_student = {}
    skill_ids = {}
    num_skill = 0

    for seq_idx, seq in enumerate(students_seq):
        for (skill, answer) in seq:
            if seq_idx not in seqs_by_student:
                seqs_by_student[seq_idx] = []
            if skill not in skill_ids:
                skill_ids[skill] = num_skill
                num_skill += 1

            seqs_by_student[seq_idx].append((skill_ids[skill], answer))

    seqs_list = list(seqs_by_student.values())

    return seqs_list, num_skill


# This class is responsible for feeding the data into the model following a specific format.
class DataGenerator(object):
    def __init__(self, features, labels, num_skills, batch_size):
        self.features = features
        self.labels = labels
        self.num_skills = num_skills
        self.batch_size = batch_size

        self.step = 0
        self.done = False
        self.feature_dim = num_skills * 2
        self.label_dim = num_skills + 1
        self.features_len = len(features)
        self.total_steps = int(math.ceil(float(self.features_len) / self.batch_size))
        self.feature_encoder = OneHotEncoder(self.feature_dim, sparse=False)
        self.label_encoder = OneHotEncoder(self.label_dim, sparse=False)

    # Ref: https://groups.google.com/forum/#!msg/keras-users/7sw0kvhDqCw/QmDMX952tq8J
    def __pad_sequences(self, sequences, maxlen=None, dim=1, dtype='int32', padding='pre', truncating='pre', value=0.):
        '''
            Override keras method to allow multiple feature dimensions.
            @dim: input feature dimension (number of features per timestep)
        '''
        lengths = [len(s) for s in sequences]

        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        x = (np.ones((nb_samples, maxlen, dim)) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understood" % padding)

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError("Padding type '%s' not understood" % padding)
        return x

    def next_batch(self):
        def fill_batches(x, y):
            for e in range(self.batch_size - len(x)):
                x.append([np.array([-1.0 for _ in range(0, self.feature_dim)])])
                y.append([np.array([-1.0 for _ in range(0, self.label_dim)])])

            return x, y

        def pad_sequences(x, y):
            max_seq_steps = max([len(seq) for seq in x])
            x = self.__pad_sequences(x, padding='pre', maxlen=max_seq_steps, dim=self.feature_dim, value=-1.0,
                                     dtype='float')
            y = self.__pad_sequences(y, padding='pre', maxlen=max_seq_steps, dim=self.label_dim, value=-1.0,
                                     dtype='float')

            return x, y

        def encode_batch(batch_questions, batch_answers):
            x = []
            y = []
            for idx, questions in enumerate(batch_questions):
                x_student = []
                y_student = []

                x_data = np.zeros(self.feature_dim, dtype=int)
                answers = batch_answers[idx]

                for skill_index, skill_value in enumerate(questions):
                    answer = answers[skill_index]

                    # Encode skill_id
                    x_student.append(x_data)
                    skill_answer = skill_value * 2 + answer
                    skill_answer = np.array([skill_answer])
                    skill_value = np.array([skill_value])
                    skill_answer = skill_answer.reshape(-1, 1)
                    skill_value = skill_value.reshape(-1, 1)

                    x_data = self.feature_encoder.fit_transform(skill_answer)[0]

                    # Encode label
                    y_data = self.label_encoder.fit_transform(skill_value)[0]
                    y_data[-1] = answer
                    y_student.append(y_data)

                x.append(x_student)
                y.append(y_student)

            return x, y

        assert (~self.done)

        start_pos = self.step * self.batch_size
        end_pos = (self.step + 1) * self.batch_size

        if end_pos >= self.features_len:
            self.done = True
            end_pos = self.features_len

        # Apply one-hot encoding
        x_batch, y_batch = encode_batch(self.features[start_pos:end_pos], self.labels[start_pos:end_pos])

        # Fill up incomplete batch
        x_batch, y_batch = fill_batches(x_batch, y_batch)

        # Pad sequences to the same size
        x_batch, y_batch = pad_sequences(x_batch, y_batch)

        self.step += 1

        return x_batch, y_batch

    def reset(self, shuffle=True):
        if shuffle:
            self.shuffle()

        self.done = False
        self.step = 0

    def shuffle(self):
        combined = list(zip(self.features, self.labels))
        random.shuffle(combined)
        self.features[:], self.labels[:] = zip(*combined)

    def get_generator(self):
        while True:
            self.reset()
            while not self.done:
                batch_features, batch_labels = self.next_batch()
                yield batch_features, batch_labels
