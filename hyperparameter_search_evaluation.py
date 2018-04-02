"""Evaluate the results of hyperparameter search and output to .csv file"""

import numpy as np
from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(420)

import itertools
from data_preprocessing import get_input_data
from cnn import get_model
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    X, y = get_input_data(train_file_path='train.json')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
