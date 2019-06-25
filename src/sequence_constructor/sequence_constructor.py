import random
import numpy as np


class SequenceConstructor:
    def __init__(self, model, exercise_ids, answers, exercises):
        self.model = model
        self.exercise_ids = exercise_ids
        self.answers = answers
        self.exercises_solved = []
        self.exercises = exercises
        self.probability_of_solving = 0.8
        self.skills_vector = []
        self.max_score = -float('inf')
        self.exercises_path = []
        self.evaluating_methods = {"sum_of_distances": self.sum_of_distances,
                                   "sum_of_probabilities": self.sum_of_probabilities}

    def get_initial_skill_vector(self):
        self.initial_skill_vector = self.model.predict_one_student(self.exercise_ids, self.answers)[0, -1, :]
        return self.initial_skill_vector


    def get_expectimax_exercise(self, depth):
        self.get_initial_skill_vector()
        score, best_exercise = self.tree_search(self.initial_skill_vector, 0, depth)
        return best_exercise

    def tree_search(self, skill_vector, exercise_id, depth):
        """
        Expectimax algorithm, with
        :param skill_vector:
        :param exercise:
        :param depth: only even number of depth
        :return:
        """
        if depth == 0:
            # Take prediction of probabilities after user attempting to solve this exercise.
            new_skill_vector = self.model.predict_one_student(self.exercise_ids, self.answers)[0, -1, :]
            # Evaluate action
            score = self.evaluate(new_skill_vector, skill_vector, metric = 'sum_of_probabilities')
            return score, exercise_id

        elif depth % 2 == 0:
            best_exercise = None
            max_score = -float('inf')
            # Loop through all exercises
            for new_exercise_id in self.exercises:
                # If the exercise is not already solved
                if new_exercise_id not in self.exercises_solved:
                    # Try next exercise
                    score, _ = self.tree_search(skill_vector, new_exercise_id, depth - 1)
                    if score > max_score:
                        max_score = score
                        best_exercise = new_exercise_id

                        print("Max score: " + "{:1.2f}".format(score) + " for exercise " + str(
                            best_exercise) + ". Depth is " + str(
                            depth) + ". Probability of solving: " + "{:1.2f}".format(skill_vector[best_exercise]))

            return max_score, best_exercise

        else:
            # Simulate that the user solves the exercise
            self.exercise_ids.append(exercise_id)
            self.exercises_solved.append(exercise_id)
            self.answers.append(1)

            # Get values for succesful sscore
            succesful_score, _ = self.tree_search(skill_vector, exercise_id, depth - 1)

            # Remove exercises from history
            del self.exercise_ids[-1]
            del self.exercises_solved[-1]
            del self.answers[-1]

            # Simulate that the user does NOT solve the exercise
            self.exercise_ids.append(exercise_id)
            self.exercises_solved.append(exercise_id)
            self.answers.append(0)

            # Get values for unseccesful score
            unsuccesful_score, _ = self.tree_search(skill_vector, exercise_id, depth - 1)

            # Remove exercises from history
            del self.exercise_ids[-1]
            del self.exercises_solved[-1]
            del self.answers[-1]

            # Calculate the expected score
            expected_score = skill_vector[exercise_id] * succesful_score + (
                    1 - skill_vector[exercise_id]) * unsuccesful_score

            return expected_score, exercise_id

    def get_next_exercise(self):
        return self.exercises_path[-1]

    def construct_hot_encoded_vector(self, exercise):
        """
        Constructing the hot encoded vector for this exercise
        :param exercise:
        :return:
        """
        hot_encoded_vector = np.zeros(2 * len(self.exercises))
        hot_encoded_vector[exercise] = 1
        return hot_encoded_vector

    def evaluate(self, new_skills_vector, old_skills_vector, metric):
        # Check new_skills_vector and previous one with
        # a sequence of actions
        evaluation = self.evaluating_methods[metric](new_skills_vector, old_skills_vector)
        return evaluation

    def sum_of_distances(self, *args):
        """
        Calculates average difference in the values of probabilities. NEW VECTOR - OLD_VECTOR
        :param args:
        :return:  a float from -1 to 1, with the average change in probability
        """
        avg_difference = np.sum(np.add(args[0], -args[1])) / len(self.exercises)

        return avg_difference

    def sum_of_probabilities(self, *args):
        """
        Calculates the average probability of solving
        :param new_skills_vector:
        :return: float
        """
        score = np.sum(args[0]) / len(self.exercises)
        return score

    # TODO 1, 2, 3 find below
    #   1) Implement evaluation of the increase of the least performing exercise
    #   2) Implement method that checks that the exercise is in the history


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
