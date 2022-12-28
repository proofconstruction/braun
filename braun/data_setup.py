import io
import os
import shutil
import zipfile
from multiprocessing import Pool
from pathlib import Path

import requests

from braun.preprocessing import grayscale_and_normalize

base_path = Path("/content/")
train_img_path = base_path / "training_images/"
val_img_path = base_path / "validation_images/"
gt_img_path = base_path / "groundtruth_images/"


def download_and_extract_noisyoffice():
    # download NoisyOffice
    zip_file_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00318/NoisyOffice.zip"
    request_data = requests.get(zip_file_url, stream=True)
    noisy_zip = zipfile.ZipFile(io.BytesIO(request_data.content))
    noisy_zip.extractall(path=base_path)


def build_directories_and_copy_noisy_data():
    # create directories for storing data
    for path in [train_img_path, gt_img_path, val_img_path]:
        path.mkdir()

    # copy NoisyOffice data where we need
    cleanpath = base_path / "NoisyOffice/SimulatedNoisyOffice/clean_images_grayscale"
    trainpath = base_path / "NoisyOffice/SimulatedNoisyOffice/simulated_noisy_images_grayscale"
    valpath = base_path / "NoisyOffice/RealNoisyOffice/real_noisy_images_grayscale"

    for groundtruth_image in sorted(os.listdir(cleanpath)):
        if os.path.isfile(cleanpath / groundtruth_image):
            shutil.copyfile(cleanpath / groundtruth_image, gt_img_path / groundtruth_image)

    for training_image in sorted(os.listdir(trainpath)):
        if os.path.isfile(trainpath / training_image):
            shutil.copyfile(trainpath / training_image, train_img_path / training_image)

    for validation_image in sorted(os.listdir(valpath)):
        if os.path.isfile(valpath / validation_image):
            shutil.copyfile(valpath / validation_image, val_img_path / validation_image)


def file_setup():
    download_and_extract_noisyoffice()
    build_directories_and_copy_noisy_data()


def generate_imageset_names(groundtruth_duplication_factor: int = 4):

    # filenames lists
    training_images_names = sorted(
        [train_img_path / filename for filename in train_img_path.iterdir() if filename.is_file()],
    )

    validation_images_names = sorted(
        [val_img_path / filename for filename in val_img_path.iterdir() if filename.is_file()],
    )

    # duplicate the ground truths 4x; we only have 18 unique groundtruth images and need 72
    groundtruth_images_names = []
    for groundtruth_image in sorted(
        [gt_img_path / filename for filename in gt_img_path.iterdir() if filename.is_file()],
    ):
        for _ in range(groundtruth_duplication_factor):
            groundtruth_images_names.append(groundtruth_image)

    return training_images_names, groundtruth_images_names, validation_images_names


def generate_and_preprocess_imagesets(training_images_names, groundtruth_images_names, validation_images_names):
    # preprocessed images
    training_images = []
    groundtruth_images = []
    validation_images = []

    pool = Pool(os.cpu_count())

    for filename in groundtruth_images_names:
        groundtruth_images.append(grayscale_and_normalize(filename))

    for filename in training_images_names:
        training_images.append(grayscale_and_normalize(filename))

    for filename in validation_images_names:
        validation_images.append(grayscale_and_normalize(filename))

    return training_images, groundtruth_images, validation_images
