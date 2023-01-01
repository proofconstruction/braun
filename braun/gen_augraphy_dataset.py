import os
from multiprocessing import Pool
from pathlib import Path

import cv2
from augraphy import AugraphyPipeline
from augraphy import BadPhotoCopy
from augraphy import PencilScribbles

from braun.preprocessing import grayscale_and_normalize


def apply_pipeline(quad):
    i = 1 + (quad[0] % 4)
    pipeline = quad[1]
    input_filename = quad[2]
    output_path = quad[3]
    output_filename = output_path / f"{input_filename.stem}-{i}-augraphy.png"

    print(f"Processing {output_filename}")

    image = cv2.imread(input_filename.as_posix())
    augmented = pipeline.augment(image)["output"]

    cv2.imwrite(output_filename.as_posix(), augmented)


def generate_enumerated_quads():
    output_path = Path("/content/training_images_augraphy")
    input_path = Path("/content/groundtruth_images")
    input_filenames = []
    for name in sorted(os.listdir(input_path)):
        for _ in range(4):  # generate 4 training images for each groundtruth
            input_filenames.append(input_path / name)
    pipeline = AugraphyPipeline([], [], [BadPhotoCopy(p=0.75), PencilScribbles(p=0.75)])
    return [(i, pipeline, input_filename, output_path) for i, input_filename in enumerate(input_filenames)]


def build_training_images_augraphy():
    output_path = Path("/content/training_images_augraphy")
    if not output_path.exists():
        output_path.mkdir()

    # build the augraphy set
    quads = generate_enumerated_quads()
    process_pool = Pool(os.cpu_count())
    process_pool.map(apply_pipeline, quads)
    process_pool.close()  # wait for all the processes in the map to exit
    # then exit


def generate_training_images_augraphy():
    build_training_images_augraphy()

    output_path = Path("/content/training_images_augraphy")
    # build the image list again
    training_images_augraphy = [
        grayscale_and_normalize((output_path / filename)) for filename in sorted(os.listdir(output_path))
    ]

    return training_images_augraphy


def generate_training_images_augraphy_names():
    training_images_augraphy_dir = Path("/content/training_images_augraphy")
    training_images_augraphy_names = sorted(
        [filename for filename in training_images_augraphy_dir.iterdir() if filename.is_file()],
    )

    return training_images_augraphy_names
