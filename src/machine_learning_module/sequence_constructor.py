from machine_learning_module.dkt_plus import LSTMModel
import random
import numpy as np


class SequenceConstructor:
    def __init__(self, model, history, exercises):
        self.model = model
        self.history = history
        self.exercises = exercises
        self.probability_of_solving = 0.8
        self.skills_vector = []
        self.max_score = -float('inf')
        self.exercises_path = []
        self.evaluating_methods = {"sum_of_distances": self.sum_of_distances}

    def get_initial_skill_vector(self):
        self.initial_skill_vector = self.model.predict(self.history)
        return self.initial_skill_vector

    def greedy_search(self):
        total_score = 0
        best_exercise = None
        # Loop through all exercises
        for exercise in self.exercises:
            # If the exercise is not already solved
            if exercise not in self.history:
                # Simulate that the user solves the exercise
                self.history.append(self.solve_exercise(exercise))

                # Take prediction of probabilities after user solving this exercise.
                successful_skill_vector = self.model.predict(self.history)

                # Evaluate action
                succesful_score = self.evaluate(successful_skill_vector, 'sum_of_distances')

                # Simulate that the user does NOT solve the exercise
                self.history[-1] = self.fail_exercise(exercise)

                # Take prediction of probabilities after user failed this exercise.
                unsuccessful_skill_vector = self.model.predict(self.history)

                # Evaluate action
                unsuccesful_score = self.evaluate(unsuccessful_skill_vector, 'sum_of_distances')

                # Total score
                total_score = self.initial_skill_vector[exercise] * succesful_score + (
                        1 - self.initial_skill_vector[exercise]) * unsuccesful_score

                # Decrease the history array
                del self.history[-1]

            if total_score > self.max_score:
                # If this score is bigger than the max save it
                self.max_score = total_score

                best_exercise = exercise

        return best_exercise

    def tree_search(self, history, skill_vector, depth, is_solved = True):
        initial_skill_vector = skill_vector

        # Check if the node is a chance node, even nodes are chance nodes, odd nodes are evaluation nodes
        if depth % 2 == 1:
            succesful_score = self.tree_search(history, skill_vector, depth - 1, is_solved = True)
            unsuccesful_score = self.tree_search(history, skill_vector, depth - 1, is_solved = False)

        else:
            # Loop through all exercises
            for exercise in self.exercises:
                # If the exercise is not already solved
                if exercise not in self.history:
            # Simulate that the user solves the exercise




    def get_next_exercise(self):
        return self.exercises_path[-1]

    def construct_hot_encoded_vector(self, exercise):
        hot_encoded_vector = np.zeros(2 * len(exercises))
        hot_encoded_vector[exercise] = 1
        return hot_encoded_vector

    def solve_exercise(self, exercise):
        hot_encoded_vector = np.zeros(2 * len(exercises))
        hot_encoded_vector[exercise] = 1
        hot_encoded_vector[2 * exercise] = 1
        return hot_encoded_vector

    def fail_exercise(self, exercise):
        hot_encoded_vector = np.zeros(2 * len(exercises))
        hot_encoded_vector[exercise] = 1
        hot_encoded_vector[2 * exercise] = 0
        return hot_encoded_vector

    def evaluate(self, new_skills_vector, old_skills_vector, metric):
        # Check new_skills_vector and previous one with
        # a sequence of actions

        return self.evaluating_methods[metric](new_skills_vector, old_skills_vector)

    def sum_of_distances(self, new_skills_vector, old_skill_vector):
        """
        Calculates average difference in the values of probabilities
        :param new_skills_vector:
        :param old_skill_vector:
        :return: a float from -1 to 1, with the average change in probability
        """
        return np.sum(new_skills_vector, -old_skill_vector) / len(self.exercises)

    # TODO Implement evaluation of the increase of the least performing exercise


if __name__ == "__main__":
    # Empty history
    history = {}

    # Exercises that we will get them from somewhere.
    exercises = []

    # Initiate an trained LSTMModel
    lstm = LSTMModel()
    lstm.load_model("something")

    # Initiate a SequenceConstructor Object
    sc = SequenceConstructor(lstm, history, exercises)
    sc.get_next_exercise()
