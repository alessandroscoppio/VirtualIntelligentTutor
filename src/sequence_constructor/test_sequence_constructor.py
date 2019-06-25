from sequence_constructor import SequenceConstructor
from src.machine_learning_module.DKT_plus.mainDKT_plus import build_model
import numpy as np
import matplotlib.pyplot as plt

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


num_of_exercises = 1377
exercises = range(num_of_exercises)
path_to_model = "../machine_learning_module/DKT_plus/cropped_hackerrank/checkpoints/n200.lo0.1.lw10.003.lw23.0/run_1/LSTM-200"
exercise_ids = ([200] * 10)
exercise_ids.extend(([400] * 10))
exercise_ids.extend(([800] * 10))
answers = ([0] * 30)
idxs = [9, 19, 29]
answers = np.array(answers)
answers[idxs] = 1
print(answers)
answers = answers.tolist()
depth = 2
best_exercise = 1286

model, sess = build_model(path_to_model, num_of_exercises)

sequence_constructor = SequenceConstructor(model, exercise_ids, answers, exercises)
# best_greedy_search_result = sequence_constructor.greedy_search()
# print(best_greedy_search_result)
initial_skill_vector = sequence_constructor.get_initial_skill_vector()
# score, best_exercise = sequence_constructor.tree_search(sequence_constructor.initial_skill_vector, 0, depth)

exercise_ids.append(best_exercise)
answers.append(1)
dkt_fig = model.plot_output_layer(exercise_ids, answers)
figure = dkt_fig.get_figure()
plt.show()
sess.close()
