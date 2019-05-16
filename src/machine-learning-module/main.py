import numpy as np
from keras.utils import to_categorical

n_exercise = 10
n_interactions = 20

interactions = np.arange(n_interactions)

print(f"Exercise: {n_exercise}\nInteractions: {n_interactions}")
print(interactions)

one_hot_encoded = to_categorical(interactions)

print(f"One hot encoding: {one_hot_encoded}")
