import numpy as np


class test_model():

    def __init__(self, num_of_exercises):
        self.num_of_exercises = num_of_exercises

    def construct_history(self, num_of_submissions):
        history = []
        for i in range(num_of_submissions):
            id = np.random.randint(self.num_of_exercises)
            has_solved = np.random.randint(0, 2)
            hot_encoded_vector = np.zeros(self.num_of_exercises * 2)
            hot_encoded_vector[id] = 1
            hot_encoded_vector[id + self.num_of_exercises] = has_solved
            history.append(hot_encoded_vector)
        return history

    def generate_random_vector(self):
        return np.random.random(self.num_of_exercises)

    def predict(self, history):
        return self.generate_random_vector()

# plot_two_skill_vectors(num_of_exercises, **results)
