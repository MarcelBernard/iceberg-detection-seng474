"""Combine outputs from multiple models"""

import json
import os

from cnn import get_model
from data_preprocessing import get_input_data, get_rotated_images
import numpy as np
from numpy.random import seed
seed(7)
from tensorflow import set_random_seed
set_random_seed(420)

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout, Input, Flatten, Activation, BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.utils import np_utils

from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Average, Dropout
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.datasets import cifar10


def get_model_A(model_input, learning_rate, dropout):
    model = (Convolution2D(64, (3, 3), strides = (1,1), activation = 'relu', input_shape = (75, 75, 2)))(model_input)
    model = (MaxPooling2D(pool_size=(3,3), strides=(2, 2)))(model)
    model = (Dropout(dropout))(model)
    # Conv2D layer 2
    model = (Convolution2D(128, (3, 3), activation='relu'))(model)
    model = (MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(model)
    model = (Dropout(dropout))(model)
    # Conv2D layer 3
    model = (Convolution2D(128, (3, 3), activation='relu'))(model)
    model = (MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(model)
    model = (Dropout(dropout))(model)
    # Conv2D layer 4
    model = (Convolution2D(64, kernel_size=(3, 3), activation='relu'))(model)
    model = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(model)
    model = (Dropout(dropout))(model)
    # Flatten data fro dense layers
    model = (Flatten())(model)
    # Dense layer 1
    model = (Dense(512, activation='relu'))(model)
    model = (Dropout(dropout))(model)
    # Dense layer 2
    model = (Dense(256, activation='relu'))(model)
    model = (Dropout(dropout))(model)
    # Dense layer 3/sigmoid boi
    model = (Dense(1, activation='sigmoid'))(model)
    # Compile model

    optimizer = optimizers.Adam(lr=learning_rate, decay=0.0)
    # model.compile(loss='binary_crossentropy',
    #               optimizer=optimizer,
    #               metrics=['accuracy'])
    model = Model(model_input, model, name='model_A')
    return model


def get_model_B(model_input, learning_rate, dropout):
    model = (Convolution2D(64, (5, 5), strides=(1, 1), activation='relu', input_shape=(75, 75, 2)))(model_input)
    model = (Convolution2D(64, (5, 5), strides=(1, 1), activation='relu', input_shape=(75, 75, 2)))(model)
    model = (Convolution2D(64, (5, 5), strides=(1, 1), activation='relu', input_shape=(75, 75, 2)))(model)
    model = (MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(model)
    model = (Dropout(dropout))(model)
    model = (Convolution2D(64, (5, 5), strides=(1, 1), activation='relu', input_shape=(75, 75, 2)))(model)
    model = (MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(model)
    model = (Dropout(dropout))(model)
    model = (Convolution2D(64, (5, 5), strides=(1, 1), activation='relu', input_shape=(75, 75, 2)))(model)
    model = (MaxPooling2D(pool_size=(2, 2), strides=(8, 8)))(model)
    model = (Dropout(dropout))(model)
    model = (Flatten())(model)
    model = (Dense(256, activation='relu'))(model)
    model = (Dropout(dropout))(model)
    model = (Dense(128, activation='relu'))(model)
    model = (Dropout(dropout))(model)
    model = (Dense(1, activation='sigmoid'))(model)
    model = Model(model_input, model, name='model_B')
    return model


def get_model_C(model_input, learning_rate, dropout):
    model = (Convolution2D(256, (3, 3), strides=(1, 1), activation='relu', input_shape=(75, 75, 2)))(model_input)
    model = (Convolution2D(256, (3, 3), strides=(1, 1), activation='relu', input_shape=(75, 75, 2)))(model)
    model = (Convolution2D(256, (3, 3), strides=(1, 1), activation='relu', input_shape=(75, 75, 2)))(model)
    model = (MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(model)
    model = (Dropout(dropout))(model)
    model = (Convolution2D(64, (3, 3), strides=(1, 1), activation='relu', input_shape=(75, 75, 2)))(model)
    model = (MaxPooling2D(pool_size=(2,2), strides=(2, 2)))(model)
    model = (Dropout(dropout))(model)
    model = (Convolution2D(32, (3, 3), strides=(1, 1), activation='relu', input_shape=(75, 75, 2)))(model)
    model = (MaxPooling2D(pool_size=(2, 2), strides=(8, 8)))(model)
    model = (Dropout(dropout))(model)
    model = (Flatten())(model)
    model = (Dense(256, activation='relu'))(model)
    model = (Dropout(dropout))(model)
    model = (Dense(128, activation='relu'))(model)
    model = (Dropout(dropout))(model)
    model = (Dense(1, activation='sigmoid'))(model)
    model = Model(model_input, model, name='model_B')
    return model


if __name__ == '__main__':
    X, y = get_input_data(train_file_path='train.json')

    # Incorporate rotated images into training data (note that this significantly increases training time)
    X_rotated, y_rotated = get_rotated_images(X, y)
    X = np.concatenate([X, X_rotated])
    y = np.concatenate([y, y_rotated])

    accuracies = []
    losses = []
    precision_scores = []
    recall_scores = []

    # Perform 10-fold cross validation
    kfolds = StratifiedKFold(n_splits=2, shuffle=True, random_state=7)
    for train_index, test_index in kfolds.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        input_shape = X_train[0, :, :, :].shape
        model_input = Input(shape=input_shape)

        model_A = get_model_A(model_input, learning_rate=0.001, dropout=0.2)
        model_B = get_model_B(model_input, learning_rate=0.001, dropout=0.2)
        model_C = get_model_C(model_input, learning_rate=0.001, dropout=0.2)

        optimizer = optimizers.Adam(lr=0.001, decay=0.0)
        model_A.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
        model_B.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
        model_C.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])

        # Train and test model
        model_A.fit(X_train, y_train, epochs=1, verbose=1)
        model_B.fit(X_train, y_train, epochs=1, verbose=1)
        model_C.fit(X_train, y_train, epochs=1, verbose=1)

        y_predictions_A = model_A.predict(X_test)
        y_predictions_B = model_B.predict(X_test)
        y_predictions_C = model_C.predict(X_test)

        y_predictions = np.average([y_predictions_A, y_predictions_B, y_predictions_C], axis=0)
        
        # Convert predictions to binary
        threshold = 0.5
        threshold_predictions = []
        for prediction in y_predictions:
            if prediction > threshold:
                threshold_predictions.append(1)
            else:
                threshold_predictions.append(0)

        # Evaluate metrics
        accuracies.append(accuracy_score(y_true=y_test, y_pred=threshold_predictions))

        loss_a = model_A.evaluate(X_test, y_test)[0]
        loss_b = model_B.evaluate(X_test, y_test)[0]
        loss_c = model_C.evaluate(X_test, y_test)[0]
        loss_avg = np.mean([loss_a, loss_b, loss_c])

        losses.append(loss_avg)
        precision_scores.append(precision_score(y_true=y_test, y_pred=threshold_predictions))
        recall_scores.append(recall_score(y_true=y_test, y_pred=threshold_predictions))

    print('Average accuracy: {}'.format(np.mean(accuracies)))
    print('Average log loss: {}'.format(np.mean(losses)))
    print('Average precision: {}'.format(np.mean(precision_scores)))
    print('Average recall: {}'.format(np.mean(recall_scores)))