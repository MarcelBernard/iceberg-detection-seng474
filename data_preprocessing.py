"""Module for preprocessing testing and training data into a format usable by the classifier."""

import json

import numpy as np


def get_input_data(train_file_path='train.json'):
    """Retrieves training (X) and label (y) matrices. Note that this can take a few seconds to run.

    Args:
        train_file_path is the path of the file containing training data.

    Returns:
        A tuple containing the X training matrix in the first position, and the y label matrix in the second position.
        X is of shape (N, 75, 75, 3), where N is the number of training images, 75 x 75 is the dimension of the images,
        and 3 represents the number of channels for each image.
    """

    with open(train_file_path, 'r') as train_file:
        json_data = train_file.read()
        train_data = json.loads(json_data)

    band_1 = [instance['band_1'] for instance in train_data]
    band_2 = [instance['band_2'] for instance in train_data]

    band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in band_1])
    band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in band_2])

    # Use the average of both bands for a third channel, since keras expects a three channel image
    band_avg = (band_1 + band_2) / 2

    # Combine all three channels into an array of 1604 tensors (number of training images) with dimension 75 x 75 x 3
    X_train = np.concatenate([band_1[:, :, :, np.newaxis], band_2[:, :, :, np.newaxis],
                             band_avg[:, :, :, np.newaxis]], axis=-1)

    # True labels of data, either iceberg or not iceberg
    y_train = np.array([instance['is_iceberg'] for instance in train_data])

    return X_train, y_train


if __name__ == '__main__':
    # Example usage
    X_train, y_train = get_input_data(train_file_path='train.json')
    print('Shape of training data is {}'.format(X_train.shape))
    print('Shape of training labels is {}'.format(y_train.shape))



