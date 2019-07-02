from machine_learning_module.DKT_plus.mainDKT_plus import build_model
from sequence_constructor.sequence_constructor import SequenceConstructor
import sys

MODELS = {
    "DKT": "./machine_learning_module/DKT_plus/cropped_hackerrank/checkpoints/n200.lo0.0.lw10.0.lw20.0/run_1/LSTM-200/LSTM-200",
    "DKT+": "./machine_learning_module/DKT_plus/cropped_hackerrank/checkpoints/n200.lo0.1.lw10.003.lw23.0/LSTM-200/LSTM-200"}


class IntelligentTutor:

    def __init__(self, machine_learning_module, sequence_constructor):
        self.machine_learning_module = machine_learning_module
        self.sequence_constructor = sequence_constructor

    def get_next_exercise(self):
        pass


if __name__ == "__main__":

    # Initial exercises
    initial_exercises = [0, 1, 2, 3, 4, 5]
    exercise_ids = []
    answers = []
    num_of_exercises = 1377
    for i in initial_exercises:
        print("Solve exercise {0}".format(i))
        exercise_ids.append(i)
        answers.append(input("Answer: "))

    # Initiate tutor
    model, session = build_model(MODELS["DKT+"], num_of_exercises)
    sequence_constructor = SequenceConstructor(model, exercise_ids, answers, range(num_of_exercises))

    ITS = IntelligentTutor(model, sequence_constructor)

    # Give exercises for ever
    try:
        while True:
            exercise_id = ITS.get_next_exercise()
            exercise_ids.append(exercise_id)
            answers.append(input("Answer: "))
    except KeyboardInterrupt:
        session.close()
        sys.exit()
