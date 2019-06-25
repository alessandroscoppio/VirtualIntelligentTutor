import os
import csv
import numpy as np
from sklearn.utils import shuffle
from src.machine_learning_module.Utils import read_file, split_tuple, split_tuple_diff
import random


def pad(data, target_length, target_value=0):
    return np.pad(data, (0, target_length - len(data)), 'constant', constant_values=target_value)


def check_tuples(to_check):
    for row in to_check:
        if len(row[1]) != len(row[2]):
            print("Problem: size mismatch")
        else:
            if len(row[1]) == row[0]:
                print('OK!')


def one_hot(indices, depth):
    encoding = np.concatenate((np.eye(depth), [np.zeros(depth)]))
    return encoding[indices]


def one_hot_diff(difficu_seqs_pad, depth):
    binned = np.digitize(difficu_seqs_pad, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    binned_oh = one_hot(binned, 10)
    return binned_oh


class OriginalInputProcessor(object):
    def process_problems_and_corrects(self, problem_seqs, correct_seqs, difficu_seqs, num_problems, is_train=True):
        """
        This function aims to process the problem sequence and the correct sequence into a DKT feedable X and y.
        :param problem_seqs: it is in shape [batch_size, None]
        :param correct_seqs: it is the same shape as problem_seqs
        :param difficu_seqs: it is the same shape as problem_seqs
        :return:
        """
        # pad the sequence with the maximum sequence length
        max_seq_length = max([len(problem) for problem in problem_seqs])
        problem_seqs_pad = np.array([pad(problem, max_seq_length, target_value=-1) for problem in problem_seqs])
        correct_seqs_pad = np.array([pad(correct, max_seq_length, target_value=-1) for correct in correct_seqs])
        try:
            difficu_seqs_pad = np.array([pad(difficu, max_seq_length, target_value=0) for difficu in difficu_seqs])
        except ValueError:
            print("sth went wrong with difficulties, clipping size")
            for idx, seq in enumerate(difficu_seqs):
                problem_seq_len = len(problem_seqs[idx])
                if len(seq) > problem_seq_len:
                    difficu_seqs[idx] = seq[0:problem_seq_len]
            difficu_seqs_pad = np.array([pad(difficu, max_seq_length, target_value=0) for difficu in difficu_seqs])

        # find the correct seqs matrix as the following way:
        # Let problem_seq = [1,3,2,-1,-1] as a and correct_seq = [1,0,1,-1,-1] as b, which are padded already
        # First, find the element-wise multiplication of a*b*b = [1,0,2,-1,-1]
        # Then, for any values 0, assign it to -1 in the vector = [1,-1,2,-1,-1] as c
        # Such that when we one hot encoding the vector c, it will results a zero vector
        temp = problem_seqs_pad * correct_seqs_pad * correct_seqs_pad  # temp is c in the comment.
        temp[temp == 0] = -1
        correct_seqs_pad = temp

        # one hot encode the information
        problem_seqs_oh = one_hot(problem_seqs_pad, depth=num_problems)
        correct_seqs_oh = one_hot(correct_seqs_pad, depth=num_problems)
        difficu_seqs_oh = one_hot_diff(difficu_seqs_pad, depth=10)

        # slice out the x and y
        if is_train:
            x_problem_seqs = problem_seqs_oh[:, :-1]
            x_correct_seqs = correct_seqs_oh[:, :-1]
            x_difficu_seqs = difficu_seqs_oh[:, :-1]
            y_problem_seqs = problem_seqs_oh[:, 1:]
            y_correct_seqs = correct_seqs_oh[:, 1:]
        else:
            x_problem_seqs = problem_seqs_oh[:, :]
            x_correct_seqs = correct_seqs_oh[:, :]
            x_difficu_seqs = difficu_seqs_oh[:, :]
            y_problem_seqs = problem_seqs_oh[:, :]
            y_correct_seqs = correct_seqs_oh[:, :]

        X = np.concatenate((x_problem_seqs, x_correct_seqs), axis=2)
        X = np.dstack((X, x_difficu_seqs))
        # = np.dstack((X, x_difficu_seqs))

        result = (X, y_problem_seqs, y_correct_seqs)
        return result


class BatchGenerator:
    """
    Generate batch for DKT model
    """

    def __init__(self, problem_seqs, correct_seqs, difficu_seqs, num_problems, batch_size,
                 input_processor=OriginalInputProcessor(),
                 **kwargs):
        self.cursor = 0  # point to the current batch index
        self.problem_seqs = problem_seqs
        self.correct_seqs = correct_seqs
        self.difficu_seqs = difficu_seqs
        self.batch_size = batch_size
        self.num_problems = num_problems
        self.num_samples = len(problem_seqs)
        self.num_batches = len(problem_seqs) // batch_size + 1
        self.input_processor = input_processor
        self._current_batch = None

    def next_batch(self, is_train=True):
        start_idx = self.cursor * self.batch_size
        end_idx = min((self.cursor + 1) * self.batch_size, self.num_samples)
        problem_seqs = self.problem_seqs[start_idx:end_idx]
        correct_seqs = self.correct_seqs[start_idx:end_idx]
        difficu_seqs = self.difficu_seqs[start_idx:end_idx]

        # x_problem_seqs, x_correct_seqs, y_problem_seqs, y_correct_seqs
        self._current_batch = self.input_processor.process_problems_and_corrects(problem_seqs,
                                                                                 correct_seqs,
                                                                                 difficu_seqs,
                                                                                 self.num_problems,
                                                                                 is_train=is_train)
        self._update_cursor()
        return self._current_batch

    @property
    def current_batch(self):
        if self._current_batch is None:
            print("Current batch is None.")
        return None

    def _update_cursor(self):
        self.cursor = (self.cursor + 1) % self.num_batches

    def reset_cursor(self):
        self.cursor = 0

    def shuffle(self):
        self.problem_seqs, self.correct_seqs = shuffle(self.problem_seqs, self.correct_seqs, random_state=42)


def read_old_format_data(filename):
    raw_data, num_problems = read_file(filename)
    X, y, diff = split_tuple_diff([value for idx, value in enumerate(raw_data)])

    max_seq_length = 0
    tuples = []
    skipped_students = 0
    for i in range(0, len(X)):

        # only keep student with at least 3 records.
        seq_length = len(X[i])
        if seq_length < 3:
            skipped_students += 1
            continue

        if len(diff[i]) != seq_length:
            continue

        new_student = (seq_length, X[i], y[i], diff[i])
        tuples.append(new_student)

        if max_seq_length < seq_length:
            max_seq_length = seq_length

    print("max_num_problems_answered:", max_seq_length)
    print("number of students with less that 3 records:", skipped_students)
    print("num_problems:", num_problems)
    print("The number of students is {0}".format(len(tuples)))
    print("Finish reading data.")

    return tuples, num_problems, max_seq_length


def read_data_from_csv(filename):
    # read the csv file
    rows = []
    with open(filename, 'r') as f:
        print("Reading {0}".format(filename))
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            rows.append(row)
        print("{0} lines was read".format(len(rows)))

    # tuples stores the student answering sequence as
    # ([num_problems_answered], [problem_ids], [is_corrects], [difficulties])
    max_seq_length = 0
    num_problems = 0
    tuples = []
    for i in range(0, len(rows), 3):
        # numbers of problem a student answered
        seq_length = int(rows[i][0])

        # only keep student with at least 3 records.
        if seq_length < 3:
            continue

        problem_seq = rows[i + 1]
        correct_seq = rows[i + 2]
        difficu_seq = rows[i + 3]

        invalid_ids_loc = [i for i, pid in enumerate(problem_seq) if pid == '']
        for invalid_loc in invalid_ids_loc:
            del problem_seq[invalid_loc]
            del correct_seq[invalid_loc]

        # convert the sequence from string to int.
        problem_seq = list(map(int, problem_seq))
        correct_seq = list(map(int, correct_seq))

        tup = (seq_length, problem_seq, correct_seq)
        tuples.append(tup)

        if max_seq_length < seq_length:
            max_seq_length = seq_length

        pid = max(int(pid) for pid in problem_seq if pid != '')
        if num_problems < pid:
            num_problems = pid
    # add 1 to num_problems because 0 is in the pid
    num_problems += 1

    print("max_num_problems_answered:", max_seq_length)
    print("num_problems:", num_problems)
    print("The number of students is {0}".format(len(tuples)))
    print("Finish reading data.")

    return tuples, num_problems, max_seq_length


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


def test_train_split(tuples, testing_rate, shuffle=True):
    seqs = tuples
    if shuffle:
        random.shuffle(seqs)

    test_idx = random.sample(range(0, len(seqs) - 1), int(len(seqs) * testing_rate))
    students_test, students_train = [], []
    max_seq_length_train = 0
    max_seq_length_test = 0

    for idx, value in enumerate(seqs):

        if idx in test_idx:
            students_test.append(value)

            if len(value[1]) > max_seq_length_test:
                max_seq_length_test = len(value[1])
            # Check if difficulties size gets messed up
            if len(value[1]) != len(value[2]):
                print("Something went wrong, submission features mismatch!")

        else:
            students_train.append(value)

            if len(value[1]) > max_seq_length_train:
                max_seq_length_train = len(value[1])

            # Check if difficulties size gets messed up
            if len(value[1]) != len(value[2]):
                print("Something went wrong, submission features mismatch!")

    return students_test, students_train, max_seq_length_test, max_seq_length_train


class DKTData:
    def __init__(self, train_path, test_path, batch_size=32):

        # Temporary solution when we have the same file for train and test
        if train_path == test_path:
            tuples, self.num_problems, self.max_seq_length = read_old_format_data(train_path)
            self.students_test, self.students_train, max_seq_length_test, max_seq_length_train = test_train_split(
                tuples, 0.2)
        else:
            self.students_train, num_problems_train, max_seq_length_train = read_data_from_csv(train_path)
            self.students_test, num_problems_test, max_seq_length_test = read_data_from_csv(test_path)
            self.num_problems = max(num_problems_test, num_problems_train)
            self.max_seq_length = max(max_seq_length_train, max_seq_length_test)

        problem_seqs = [student[1] for student in self.students_train]
        correct_seqs = [student[2] for student in self.students_train]
        difficu_seqs = [student[3] for student in self.students_train]

        self.train = BatchGenerator(problem_seqs, correct_seqs, difficu_seqs, self.num_problems, batch_size)

        problem_seqs = [student[1] for student in self.students_test]
        correct_seqs = [student[2] for student in self.students_test]
        difficu_seqs = [student[3] for student in self.students_test]

        self.test = BatchGenerator(problem_seqs, correct_seqs, difficu_seqs, self.num_problems, batch_size)
