import random
import numpy as np
from src.machine_learning_module.DKT_plus.mainDKT_plus import build_model

class Experimentator():

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def predict_one_student(self, exercise_sequence, correct_sequence):
        results = self.model.predict(exercise_sequence, correct_sequence)
        return results

    def build_testing_batch(self, seq_list, correct_list, batch_size, num_of_exercises):
        """
        Extend a sequence for 1 student to a batch of random sequences for a testing purposes
        :param seq_list:
        :param batch_size:
        :param num_of_exercises:
        :return:
        """

        max_length = len(seq_list)
        final_batch = []
        answers_batch = []

        for i in range(batch_size - 1):

            sum = 0
            exercies_sequence = []
            answers_sequence = []
            sequence_len = np.random.randint(3, max_length)

            while sum < sequence_len - 1:

                num_of_attempts = np.random.randint(1, sequence_len - sum)
                exercise = np.random.randint(0, num_of_exercises)
                attempts = [exercise] * num_of_attempts
                exercies_sequence.extend(attempts)
                sum += num_of_attempts

            final_batch.append(exercies_sequence)
            answers_batch.append([0] * len(exercies_sequence))

        final_batch.append(seq_list)
        answers_batch.append(correct_list)

        return final_batch, answers_batch



test_ex_list = [8] * 20
test_ex_list.extend([10]*20)
test_answer_list = [0] * 40
test_answer_list[19] = 1
test_answer_list[18] = 1
test_answer_list[17] = 1


model, sess = build_model(1377)

experimentator = Experimentator(model, None)
batch, answers = experimentator.build_testing_batch(test_ex_list, test_answer_list, 5, 100)
result = model.predict_one_student(batch, answers)

print()
sess.close()
