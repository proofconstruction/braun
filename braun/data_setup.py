import io
import os
import shutil
import zipfile
from multiprocessing import Pool
from pathlib import Path

import requests

from preprocessing import grayscale_and_normalize


def download_and_extract_noisyoffice():
    # download NoisyOffice
    zip_file_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00318/NoisyOffice.zip"
    extract_dir = "/content/NoisyOffice"
    request_data = requests.get(zip_file_url, stream=True)
    noisy_zip = zipfile.ZipFile(io.BytesIO(request_data.content))
    noisy_zip.extractall(path=extract_dir)


def build_directories_and_copy_noisy_data():
    # create directories for storing data
    base_dir = Path("/content/")

    training_images_dir = base_dir / "training_images"
    groundtruth_images_dir = base_dir / "groundtruth_images"
    validation_images_dir = base_dir / "validation_images"

    for path in [training_images_dir, groundtruth_images_dir, validation_images_dir]:
        path.mkdir()

    # copy NoisyOffice data where we need
    cleanpath = base_dir / "/NoisyOffice/SimulatedNoisyOffice/clean_images_grayscale"
    trainpath = base_dir / "/NoisyOffice/SimulatedNoisyOffice/simulated_noisy_images_grayscale"
    valpath = base_dir / "/NoisyOffice/RealNoisyOffice/real_noisy_images_grayscale"

    for groundtruth_image in sorted(os.listdir(cleanpath)):
        shutil.copyfile(groundtruth_image, groundtruth_images_dir)

    for training_image in sorted(os.listdir(trainpath)):
        shutil.copyfile(training_image, training_images_dir)

    for validation_image in sorted(os.listdir(valpath)):
        shutil.copyfile(validation_image, validation_images_dir)


def file_setup():
    download_and_extract_noisyoffice()
    build_directories_and_copy_noisy_data()


def generate_imageset_names(
    training_images_dir: Path,
    groundtruth_images_dir: Path,
    validation_images_dir: Path,
    groundtruth_duplication_factor: int = 4,
):
    # filenames lists
    training_images_names = sorted(
        [
            "/content/training_images/" + filename
            for filename in Path(training_images_dir).iterdir()
            if filename.is_file()
        ],
    )

    groundtruth_images_names = []

    validation_images_names = sorted(
        [
            "/content/validation_images/" + filename
            for filename in Path(validation_images_dir).iterdir()
            if filename.is_file()
        ],
    )

    # duplicate the ground truths 4x; we only have 18 unique groundtruth images and need 72
    for groundtruth_image in sorted(
        [
            "/content/groundtruth_images/" + filename
            for filename in Path(groundtruth_images_dir).iterdir()
            if filename.is_file()
        ],
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
