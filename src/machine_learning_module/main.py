# from VirtualIntelligentTutor.src.machine_learning_module.dkt_plus import LSTMModel
# from VirtualIntelligentTutor.src.machine_learning_module.Utils import *

''''''
from src.machine_learning_module.Utils import read_file, split_dataset, DataGenerator
from src.machine_learning_module.dkt_plus import LSTMModel

train_log = "logs/dktmodel.train.log"  # File to save the training log.
eval_log = "logs/dktmodel.eval.log"  # File to save the testing log.
optimizer = "adagrad"  # Optimizer to use
lstm_units = 250  # Number of LSTM units
batch_size = 20  # Batch size
epochs = 10  # Number of epochs to train
dropout_rate = 0.6  # Dropout rate
verbose = 1  # Verbose = {0,1,2}
testing_rate = 0.2  # Portion of data to be used for testing
validation_rate = 0.2  # Portion of training data to be used for validation
''''''

dataset = "data/ASSISTments_skill_builder_data.csv"  # Dataset path
dataset, num_problems = read_file(dataset)
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(dataset, validation_rate, testing_rate)

# Create generators for training/testing/validation
train_gen = DataGenerator(X_train[0:10], y_train[0:10], num_problems, batch_size)
val_gen = DataGenerator(X_val[0:10], y_val[0:10], num_problems, batch_size)
test_gen = DataGenerator(X_test[0:10], y_test[0:10], num_problems, batch_size)


ourModel = LSTMModel(batch_size=batch_size, max_n_interactions=train_gen.num_skills,
                     n_exercises=num_problems)

ourModel.fit(train_gen, val_gen, epochs=5, verbose=2)