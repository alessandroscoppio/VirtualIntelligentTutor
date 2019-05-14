from machine_learning_module.dkt_plus import LSTMModel
import random


class SequenceConstructor:
    def __init__(self, model, history, exercises):
        self.model = model
        self.history = history
        self.exercises = exercises
        self.probability_of_solving = 0.8
        self.skills_vector = []
        self.max_score = -float('inf')
        self.exercises_path = []

    def tree_search(self):
        score = 0
        # Loop through all exercises
        for exercise in self.exercises:
            # If the exercise is not already solved
            if exercise not in self.history:
                # Simulate that the user solves the exercise
                self.history[exercise] = self.solve_exercise()

                # Take prediction of probabilities after user solving this exercise.
                new_skills_vector = self.model.predict(self.skills_vector)

                # Evaluate action
                score = self.evaluate(new_skills_vector)

                # Move on to the next exercise.
                self.tree_search()

            if score > self.max_score:
                # If this score is bigger than the max save it
                self.max_score = score

                # Append good path
                self.exercises_path.append(exercise)

    def get_next_exercise(self):
        return self.exercises_path[-1]

    def solve_exercise(self):
        return random.random() < self.probability_of_solving

    def evaluate(self, new_skills_vector):
        # Check new_skills_vector and previous one with
        # a sequence of actions
        return random.random()


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
