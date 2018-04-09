import numpy as np
from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(420)

import itertools
from data_preprocessing import get_input_data
from cnn import get_model
from sklearn.model_selection import train_test_split

# Note that the order of hyperparameters in each permutation is the same as the order they appear here
learning_rates = [0.05, 0.01, 0.005, 0.001, 0.0005]
epoch_counts = [5, 10, 15]
batch_sizes = [32, 64, 128]
dropout = [0.2, 0.4, 0.6, 0.8]

hyperparams = [learning_rates, epoch_counts, batch_sizes, dropout]

permutations = list(itertools.product(*hyperparams))

marcel = permutations[:45]
mahfuza = permutations[45:90]
nigel = permutations[90:135]
lambert = permutations[135:]

X, y = get_input_data(train_file_path='train.json')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for params in permutations[45:]:
    model = get_model(learning_rate=params[0], dropout=params[3])

    accuracies = []
    losses = []
    precision_scores = []
    recall_scores = []

    # Train and test model
    model.fit(X_train, y_train, epochs=params[1], verbose=1, batch_size=params[2])
    model.save('hyperparams_{}_{}_{}_{}'.format(*params))
