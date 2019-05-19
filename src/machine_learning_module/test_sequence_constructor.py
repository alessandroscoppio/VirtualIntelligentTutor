from src.machine_learning_module.sequence_constructor import SequenceConstructor
import numpy as np


class test_model():

    def __init__(self, num_of_exercises):
        self.num_of_exercises = num_of_exercises

    def construct_history(self, num_of_submissions):
        history = []
        for i in range(num_of_submissions):
            id = np.random.randint(self.num_of_exercises)
            has_solved = np.random.randint(0, 2)
            hot_encoded_vector = np.zeros(num_of_exercises * 2)
            hot_encoded_vector[id] = 1
            hot_encoded_vector[id + self.num_of_exercises] = has_solved
            history.append(hot_encoded_vector)
        return history

    def generate_random_vector(self):
        return np.random.random(self.num_of_exercises)

    def predict(self, history):
        return self.generate_random_vector()


num_of_exercises = 10
num_of_attempts = 5
exercises = range(num_of_exercises)
model = test_model(len(exercises))
history = model.construct_history(num_of_attempts)
sequence_constructor = SequenceConstructor(model, history, exercises)
# best_greedy_search_result = sequence_constructor.greedy_search()
# print(best_greedy_search_result)
best_tree_search = sequence_constructor.get_expectimax_exercise(4)
print(best_tree_search)