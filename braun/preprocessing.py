from pathlib import Path

import cv2
import numpy as np


# A utility function for preprocessing
def grayscale_and_normalize(image_path: Path):
    # read the images in as f32 arrays, for increased precision
    img = cv2.imread(image_path.as_posix()).astype(np.float32)

    # resize to correct dimensions
    img = cv2.resize(img, (540, 420))

    # grayscale image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalize pixel intensity values to the 8-bit range (still f32)
    img = img / 255.0

    # remove the color channels from the image
    img = np.reshape(img, (420, 540, 1))

    return img
