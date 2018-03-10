import json
import os
import numpy as np
from data_preprocessing import get_input_data

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.utils import np_utils


def get_model():
    """ Generates and compiles a sequential model with 4 convolutional 2D layers,
    2 dense layer and a sigmoid function. Initial implementation inspired by
    https://www.kaggle.com/cbryant/keras-cnn-statoil-iceberg-lb-0-1995-now-0-1516"""
    # Sequential model
    model = Sequential()
    # Conv2D input layer
    model.add(Convolution2D(64, (3, 3), strides = (1,1), activation = 'relu', input_shape = (75, 75, 2)))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2, 2)))
    model.add(Dropout(0.2))
    # Conv2D layer 2
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.2))
    # Conv2D layer 3
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
    model.add(Dropout(0.2))
    # Conv2D layer 4
    model.add(Convolution2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))
    # Flatten data fro dense layers
    model.add(Flatten())
    # Dense layer 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    # Dense layer 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    # Dense layer 3/sigmoid boi
    model.add(Dense(1, activation='sigmoid'))
    # Compile model

    optimizer = optimizers.Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    X, y = get_input_data(train_file_path='train.json')
    model = get_model()

    accuracies = []
    losses = []
    precision_scores = []
    recall_scores = []

    # Perform 10-fold cross validation
    kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    for train_index, test_index in kfolds.split(X, y):
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        # Train and test model
        model.fit(X_train, y_train, epochs=10, verbose=1)
        y_predictions = model.predict(X_test)

        # Evaluate metrics
        accuracies.append(accuracy_score(y_true=y_test, y_pred=y_predictions))
        losses.append(model.evaluate(X_test, y_test))

    print('Average accuracy: {}'.format(np.mean(accuracies)))
    print('Average log loss: {}'.format(np.mean(losses)))
    print('Average precision: {}'.format(np.mean(precision_scores)))
    print('Average recall: {}'.format(np.mean(recall_scores)))
