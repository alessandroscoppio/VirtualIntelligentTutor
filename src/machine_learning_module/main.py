# from VirtualIntelligentTutor.src.machine_learning_module.dkt_plus import LSTMModel
# from VirtualIntelligentTutor.src.machine_learning_module.Utils import *


# -- ignore scikit deprecated warnings, there were too many at each step --
# https://stackoverflow.com/a/33616192/9344265
from src.machine_learning_module.DKTmodel import model_evaluate
from src.machine_learning_module.Utils import read_file, split_dataset, DataGenerator
from src.machine_learning_module.dkt_plus import LSTMModel


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
# --------------------------------------------------------------------------

# from Utils import read_file, split_dataset, DataGenerator
# from dkt_plus import LSTMModel

train_log = "logs/dktmodel.train.log"  # File to save the training log.
eval_log = "logs/dktmodel.eval.log"  # File to save the testing log.
optimizer = "adagrad"  # Optimizer to use
lstm_units = 250  # Number of LSTM units
batch_size = 20  # Batch size
epochs = 10  # Number of epochs to train
dropout_rate = 0.6  # Dropout rate
verbose = 2  # Verbose = {0,1,2}
testing_rate = 0.2  # Portion of data to be used for testing
validation_rate = 0.2  # Portion of training data to be used for validation
''''''
"""
FIRST WAY OF LOADING DATASET:
 this three lines of code
 (stolen from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/src/DKT.ipynb)
 load the csv dataset (tp download from github) and creates data generators.
 I modified my model to accept this generator and use fit_generator 
 but at some points there is something that make it break.
 (Reshape your data either using array.reshape(-1, 1) if your data has a single feature
  or array.reshape(1, -1) if it contains a single sample.)
"""

dataset = "data/ASSISTments_skill_builder_data.csv"  # Dataset path
dataset, num_problems = read_file(dataset)
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(dataset, validation_rate, testing_rate)

# Create generators for training/testing/validation
train_gen = DataGenerator(X_train[0:10], y_train[0:10], num_problems, batch_size)
val_gen = DataGenerator(X_val[0:10], y_val[0:10], num_problems, batch_size)
test_gen = DataGenerator(X_test[0:10], y_test[0:10], num_problems, batch_size)

"""
I have also tried different implementations ways to load data in this fucking module:
- https://github.com/siyuanzhao/2016-EDM
- https://github.com/davidoj/deepknowledgetracingTF

and of course dkt+, the latest and better model so far:
- https://github.com/ckyeungac/deep-knowledge-tracing-plus/blob/master/load_data.py

all of them have in common 2 things:
    - they all use some sort of custom data manager to train the network, something like DataGenerator
        or BatchGenerator or some object that deals internally with batches size, current batch, 
        crossvalidation, test and validation data, etc.
    - even if implemented in different ways and with DIFFERENT OUTPUTS sometimes,
        we want our data to look like this:
        
        INPUT:  array/tensor with dims: (batch_size, ??, 2*n_problems)
        OUTPUT: array/tensor with dims: (1, n_problems)

I'm going to sleep because I did computer vision till 6 and my head does not work anymore,
please take a look at our code, the github pages I linked and let's try to make this thing run :D    

I suggest to read documentation of:
    Logits
    Masking layer
    fit_generator / train on batch
    everything is not really clear when you see it.
    
    
    Once data is loaded in the correct format, the model should run smoothly, model layer/parameters
    have been cross checked with multiple implementation, especially Keras based ones:
    - https://github.com/CAHLR/DKT_pre
    - https://github.com/mmkhajah/dkt/blob/master/dkt.py (this seems the easier one)
    
    please have a look at them, I may have missed something important and we could still look it up to compare
"""

'''
Ibro:
I have commented out this model and put the one below,
which is https://github.com/LiangbeiXu/Deep-Knowledge-Tracing/blob/master/src/StudentModel.py
I also created folders saved_models and logs because the model below requires it.
'''
# ourModel = LSTMModel(hidden_units=200, batch_size=batch_size, n_exercises=num_problems)
# ourModel.fit(train_gen, val_gen, epochs=5, verbose=2)
# model_evaluate(test_gen, ourModel.model, metrics=['auc','acc','pre'], verbose=2)

# ------------- The other DKT model -----------------
from DKTmodel import DKTModel

# Create model
best_model_file = "saved_models/ASSISTments.best.model.weights.hdf5"  # File to save the model.

student_model = DKTModel(num_skills = train_gen.num_skills,
                         num_features = train_gen.feature_dim,
                         optimizer = optimizer,
                         hidden_units = lstm_units,
                         batch_size = batch_size,
                         dropout_rate = dropout_rate)

history = student_model.fit(train_gen,
                            epochs = epochs,
                            val_gen = val_gen,
                            verbose = verbose,
                            filepath_bestmodel = best_model_file,
                            filepath_log = train_log)
# -----------------------------------------------------
