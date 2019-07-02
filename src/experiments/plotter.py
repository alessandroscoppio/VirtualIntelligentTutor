import numpy as np
from matplotlib import pyplot as plt
from src.machine_learning_module.DKT_plus.utils import extend_sequence_to_batch


def plot_single_student_history(test_ex_list, test_answer_list, model, num_of_exercises, student_id = -1,
                                title = 'History of submissions of a student'):
    batch, answers = extend_sequence_to_batch(test_ex_list, test_answer_list, 2, 100)
    result = model.predict_one_student(batch[student_id], answers[student_id])
    evaluation = []
    colors = []
    for i in range(len(result)):
        evaluation.append(np.sum(result[i]) / num_of_exercises)
        if test_answer_list[i] == 1:
            colors.append("g")
        else:
            colors.append("r")
    y_pos = np.arange(len(evaluation))
    plt.style.use('seaborn-darkgrid')
    plt.bar(y_pos, evaluation, color = colors)

    # Make the xticks not repeatable
    xticks = [test_ex_list[0]]
    for i in range(1, len(test_ex_list)):
        if test_ex_list[i - 1] == test_ex_list[i] and i != 1:
            xticks.append("")
        else:
            xticks.append(test_ex_list[i])

    plt.xticks(y_pos, xticks, rotation = '65')
    plt.ylim(0.35, 0.6)
    plt.title(title)
    plt.xlabel('ID of the exercises')
    plt.ylabel('Evaluation of the student (AVG probability)')
    plt.show()


def plot_two_skill_vectors(num_of_exercises, **kwargs):
    """
    Plot sorted distribution of the probabilites of solving the exercises. IMPORTANT: the values on X-axis represent
     order of the exercises after it has been sorted for a given student (skill vector)
    :param num_of_exercises: total number of exercises
    :param kwargs: dictionary of skill vectors
    :return:
    """
    ax = plt.subplot(111)
    x = range(num_of_exercises)

    for key, skill_vector in kwargs.items():
        skill_vector = np.sort(skill_vector)
        ax.scatter(x, skill_vector, label = key, s = 0.1)
    plt.style.use('seaborn-darkgrid')
    plt.legend()
    plt.title("Sorted probability of solving the exercises")
    plt.show()