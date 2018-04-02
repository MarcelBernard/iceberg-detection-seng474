"""Evaluate the results of hyperparameter search and output to .csv file"""

import numpy as np
from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(420)

import csv
import keras
from data_preprocessing import get_input_data
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score

if __name__ == '__main__':
    X, y = get_input_data(train_file_path='train.json')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    saved_models = [f for f in os.listdir('.') if f.startswith('hyperparams_')]

    with open('hyperparamater_search_results.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        header = ['learning_rate', 'epochs', 'batch_size', 'dropout', 'accuracy', 'log_loss', 'precision', 'recall']
        writer.writerow(header)

    for saved_model in saved_models:
        model = keras.models.load_model(saved_model)
        y_predictions = model.predict(X_test)

        # Convert predictions to binary
        threshold = 0.5
        threshold_predictions = []
        for prediction in y_predictions:
            if prediction > threshold:
                threshold_predictions.append(1)
            else:
                threshold_predictions.append(0)

        # Evaluate metrics
        accuracy = accuracy_score(y_true=y_test, y_pred=threshold_predictions)
        loss = model.evaluate(X_test, y_test)[0]
        precision = precision_score(y_true=y_test, y_pred=threshold_predictions)
        recall = recall_score(y_true=y_test, y_pred=threshold_predictions)

        hyperparams = saved_model.split('_')
        learning_rate = hyperparams[1]
        epochs = hyperparams[2]
        batch_size = hyperparams[3]
        dropout = hyperparams[4]

        # Record results
        with open('hyperparamater_search_results.csv', 'a') as csv_file:
            writer = csv.writer(csv_file)
            row = [learning_rate, epochs, batch_size, dropout, accuracy, loss, precision, recall]
            writer.writerow(row)