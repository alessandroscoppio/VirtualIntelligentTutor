from src.machine_learning_module.DKT_plus.mainDKT_plus import build_model
from src.experiments.experimentator import Experimentator
from src.experiments.plotter import plot_two_skill_vectors

data_path = "./machine_learning_module/data/submissions_20+.csv"
trained_models = {
    "DKT": "./machine_learning_module/DKT_plus/cropped_hackerrank/checkpoints/n200.lo0.0.lw10.0.lw20.0/run_1/LSTM-200/LSTM-200",
    "DKT+": "./machine_learning_module/DKT_plus/cropped_hackerrank/checkpoints/n200.lo0.1.lw10.003.lw23.0/run_1/LSTM-200/LSTM-200"}

if __name__ == "__main__":

    # Initializing model
    num_of_exercises = 1377
    model_name = "DKT+"
    model, sess = build_model(trained_models[model_name], num_of_exercises)
    experimentator = Experimentator(model)
    students = experimentator.load_tuples(data_path)
    experimentator.load_scores()


    # Show average probability for a high performing students
    experimentator.plot_dynamics_high_performing_students(1)


    # Show average probability for a low performing students
    experimentator.plot_dynamics_low_performing_students(1)


    # Custom list of attempts and expectimax prediction
    exercise_ids = [82, 82, 83, 83, 83, 2, 3, 100, 105, 111, 112, 113, 115, 123, 123, 124]
    answers = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    student_id = -1

    results = experimentator.get_expectimax_prediction(exercise_ids, answers, num_of_exercises)
    plot_two_skill_vectors(num_of_exercises, **results)
    sess.close()
