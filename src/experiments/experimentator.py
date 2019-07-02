from machine_learning_module.DKT_plus.load_data import read_old_format_data
from machine_learning_module.DKT_plus.mainDKT_plus import build_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sequence_constructor.sequence_constructor import SequenceConstructor


class Experimentator():

    def __init__(self, model):
        self.model = model

    def predict(self, exercise_sequence, correct_sequence):
        results = self.model.predict_one_student(exercise_sequence, correct_sequence)
        return results

    def load_tuples(self, path_to_data):
        self.tuples, self.num_of_problems, _ = read_old_format_data(path_to_data)
        return self.tuples

    def calculate_students_score(self, write_csv = True):
        scores = []
        i = 0
        counter = 0
        for student in self.tuples:
            score = self.model.predict_one_student(student[1], student[2])[-1, :]
            score = np.sum(score) / self.num_of_problems
            scores.append(score)
            if i % 1000 == 0:
                print(i)
            i += 1

        keys = range(len(self.tuples))
        values = scores
        df = pd.DataFrame({'col1': keys, 'col2': values})

        if write_csv:
            df.to_csv('scores.csv', index = False)
        return scores

    def load_scores(self):
        df = pd.DataFrame.from_csv('./experiments/scores.csv')
        self.scores_dataframe = df

    def sort_students(self):
        self.scores_dataframe['ids'] = range(self.scores_dataframe.shape[0])
        self.scores_dataframe = self.scores_dataframe.sort_values('col2')
        return self.scores_dataframe

    def plot_dynamics_high_performing_students(self, rank):
        """
        Investigate the history of submissions of student ranked by given number.
        :param rank: rank of the targeted student
        :return:
        """
        self.sort_students()

        best_student_id = self.scores_dataframe.iloc[-rank, 1]
        best_student = self.tuples[best_student_id]

        plt.figure(figsize = (15, 2))
        dkt_fig = self.model.plot_output_layer(problem_seq = best_student[1], correct_seq = best_student[2],
                                          target_problem_ids = best_student[1])
        figure = dkt_fig.get_figure()
        plt.show()

    def plot_dynamics_low_performing_students(self, rank):
        """
        Investigate the history of submissions of student ranked by given number.
        :param rank: rank of the targeted student (from the bottom)
        :return:
        """
        self.sort_students()

        low_performing_student = self.scores_dataframe.iloc[-rank, 1]
        low_performing_studetn = self.tuples[low_performing_student]

        plt.figure(figsize = (15, 2))
        dkt_fig = self.model.plot_output_layer(problem_seq = low_performing_studetn[1],
                                          correct_seq = low_performing_studetn[2],
                                          target_problem_ids = low_performing_studetn[1])
        figure = dkt_fig.get_figure()
        plt.show()

    def get_expectimax_prediction(self, exercise_ids, answers, num_of_exercises, depth = 2):
        """
        Run a sequence constructor and get the results.
        :param exercise_ids: list of exercise IDs, attempts of solving
        :param answers: list of 0 and 1, indictting whether the exercises were solved or not
        :param num_of_exercises: total number of exercises
        :param depth: depth of search, must be EVEN, minimum = 2
        :return: the dictionary of skill vectors,
            {Original: skill_vector without solving suggested exercise
            Succesful submission of N: skill_vector with successful attempt of solving suggested exercise
            Unsuccesful submission of N: skill_vector with unsuccessful attempt of solving suggested exercise}
        """
        sequence_constructor = SequenceConstructor(self.model, exercise_ids, answers, num_of_exercises)
        initial_skill_vector = sequence_constructor.get_initial_skill_vector()
        score, best_exercise = sequence_constructor.tree_search(sequence_constructor.initial_skill_vector, 0, depth)
        exercise_ids.append(best_exercise)
        print("The best exercise suggested by expectimax: ", best_exercise)
        answers.append(1)
        succesful_skill_vector = self.model.predict_one_student(exercise_ids, answers)[-1, :]
        del answers[-1]
        answers.append(0)
        unsuccesful_skill_vector = self.model.predict_one_student(exercise_ids, answers)[-1, :]
        del answers[-1]
        answers.append(1)
        results = {"Original": initial_skill_vector,
                   "Succesful submission of " + str(best_exercise): succesful_skill_vector,
                   "Unsuccesful submission of " + str(best_exercise): unsuccesful_skill_vector}

        return results


def run_sequence_constructor(model, exercise_ids, answers, exercises, depth = 2):
    sequence_constructor = SequenceConstructor(model, exercise_ids, answers, exercises)
    initial_skill_vector = sequence_constructor.get_initial_skill_vector()
    score, best_exercise = sequence_constructor.tree_search(sequence_constructor.initial_skill_vector, 0, depth)
    exercise_ids.append(best_exercise)
    answers.append(1)
    succesful_skill_vector = model.predict_one_student(exercise_ids, answers)[-1, :]
    del answers[-1]
    answers.append(0)
    unsuccesful_skill_vector = model.predict_one_student(exercise_ids, answers)[-1, :]
    del answers[-1]
    answers.append(1)
    results = {"Original": initial_skill_vector,
               "Succesful submission of " + str(best_exercise): succesful_skill_vector,
               "Unsuccesful submission of " + str(best_exercise): unsuccesful_skill_vector}
    return results

#