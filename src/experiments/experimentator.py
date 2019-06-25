import random
import numpy as np
from src.machine_learning_module.DKT_plus.mainDKT_plus import build_model
import matplotlib.pyplot as plt


class Experimentator():

    def __init__(self, model):
        self.model = model

    def predict(self, exercise_sequence, correct_sequence):
        results = self.model.predict_one_student(exercise_sequence, correct_sequence)
        return results

    def build_testing_batch(self, seq_list, correct_list, batch_size, num_of_exercises):
        """
        Extend a sequence for 1 student to a batch of random sequences for a testing purposes
        :param seq_list: sequence of exercises
        :param batch_size: desired batch sice
        :param num_of_exercises: total num of exercises in dataset
        :return: batch with random history of students
        """

        max_length = len(seq_list)
        final_batch = []
        answers_batch = []

        for i in range(batch_size - 1):

            sum = 0
            exercies_sequence = []
            answers_sequence = []
            sequence_len = max_length

            while sum < sequence_len - 1:

                num_of_attempts = np.random.randint(1, sequence_len - sum)
                exercise = np.random.randint(0, num_of_exercises)
                attempts = [exercise] * num_of_attempts
                answers = [0] * num_of_attempts

                answers[-1] = np.random.choice([0, 1])
                answers_sequence.extend(answers)
                exercies_sequence.extend(attempts)
                sum += num_of_attempts

            final_batch.append(exercies_sequence)
            answers_batch.append(answers_sequence)

        final_batch.append(seq_list)
        answers_batch.append(correct_list)

        return final_batch, answers_batch


models = {
    "DKT": "../machine_learning_module/DKT_plus/cropped_hackerrank/checkpoints/n200.lo0.0.lw10.0.lw20.0/run_1/LSTM-200/LSTM-200",
    "DKT+": "../machine_learning_module/DKT_plus/cropped_hackerrank/checkpoints/DKT_regularization/LSTM-200/LSTM-200"}
num_of_exercises = 1377
model_name = "DKT"

test_ex_list = [8] * 10
test_ex_list.extend([10] * 10)
test_ex_list.extend([12] * 10)
test_answer_list = [0] * 30
test_answer_list[9] = 1
test_answer_list[8] = 1
test_answer_list[19] = 1
test_answer_list[18] = 1
test_answer_list[29] = 1
test_answer_list[28] = 1
student_id = -1

model, sess = build_model(models[model_name], num_of_exercises)
experimentator = Experimentator(model)
batch, answers = experimentator.build_testing_batch(test_ex_list, test_answer_list, 5, 100)
result = model.predict_one_student([batch[student_id]], [answers[student_id]])
plt.figure(figsize = (15, 2))
dkt_fig = model.plot_output_layer(problem_seq = batch[student_id], correct_seq = answers[student_id])
figure = dkt_fig.get_figure()
figure.savefig(model_name + 'result.pdf', bbox_inches = 'tight')  # , bbox_extra_artist=[lgd])
sess.close()

