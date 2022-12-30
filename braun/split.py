import numpy as np
from sklearn.model_selection import train_test_split


def training_split(training_images, groundtruth_images, test_size: float = 0.15):
    # convert lists to numpy.ndarrays
    training_images_arr = np.asarray(training_images)
    groundtruth_images_arr = np.asarray(groundtruth_images)

    # training_images_arr, training_images_count, groundtruth_images_arr, groundtruth_images_count
    return train_test_split(training_images_arr, groundtruth_images_arr, test_size=test_size)
