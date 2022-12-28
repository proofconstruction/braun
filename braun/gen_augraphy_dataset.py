import os
from multiprocessing import Pool
from pathlib import Path

import augraphy
import cv2
from git import Repo


def get_shabby_pipeline():
    Repo.clone_from("https://github.com/sparkfish/shabby-pages", "shabby")
    from shabby.daily_pipeline import get_pipeline


def apply_pipeline(enumerate_pair):
    i = 1 + (enumerate_pair[0] % 4)
    input_filename = enumerate_pair[1][0]
    output_path = enumerate_pair[1][1]
    output_filename = output_path / f"{input_filename.stem}-{i}-augraphy.png"

    print(f"Processing {output_filename}")

    image = cv2.imread(input_filename.as_posix())
    pipeline = get_pipeline()
    augmented = pipeline.augment(image)["output"]

    cv2.imwrite(output_filename.as_posix(), augmented)


def generate_enumerated_pairs():
    output_path = Path("/content/training_images_augraphy")
    input_path = Path("/content/groundtruth_images")
    input_filenames = []

    for name in sorted(os.listdir(input_path)):
        for _ in range(4):
            input_filenames.append(input_path / name)

    return enumerate([(input_filename, output_path) for input_filename in input_filenames])


def generate_training_images_augraphy():
    output_path = Path("/content/training_images_augraphy")

    # build the augraphy set
    enumerate_pairs = generate_enumerated_pairs()
    process_pool = Pool(os.cpu_count())
    process_pool.map(apply_pipeline, enumerate_pairs)

    # build the image list again
    training_images_augraphy = [
        cv2.imread(output_path.as_posix() + filename) for filename in sorted(os.listdir(output_path))
    ]

    return training_images_augraphy


def generate_training_images_augraphy_names():
    training_images_augraphy_dir = Path("/content/training_images_augraphy")
    training_images_augraphy_names = [
        filename for filename in training_images_augraphy_dir.iterdir() if filename.is_file()
    ]

    return training_images_augraphy_names
