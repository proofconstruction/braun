from keras.callbacks import EarlyStopping

from data_setup import file_setup
from data_setup import generate_and_preprocess_imagesets
from data_setup import generate_imageset_names
from gen_augraphy_dataset import generate_training_images_augraphy
from gen_augraphy_dataset import generate_training_images_augraphy_names
from gen_augraphy_dataset import get_shabby_pipeline
from plotting import plot_training_vs_groundtruth_images
from plotting import plot_traininginstance_loss_and_error
from split import training_split
from training import generate_training_instances

if __name__ == "__main__":

    # we want to save the models we train at the end of this
    save_models = True

    # download NoisyOffice data and put it where we need it
    file_setup()

    # import the NoisyOffice data
    training_images_names, groundtruth_images_names, validation_images_names = generate_imageset_names()
    training_images, groundtruth_images, validation_images = generate_and_preprocess_imagesets(
        training_images_names,
        groundtruth_images_names,
        validation_images_names,
    )

    # get the latest shabby pipeline
    get_shabby_pipeline()

    # generate and import Augraphy data
    training_images_augraphy = generate_training_images_augraphy()
    training_images_augraphy_names = generate_training_images_augraphy_names()

    # display NoisyOffice images
    plot_training_vs_groundtruth_images(
        training_images,
        training_images_names,
        groundtruth_images,
        groundtruth_images_names,
    )

    # display Augraphy images
    plot_training_vs_groundtruth_images(
        training_images_augraphy,
        training_images_augraphy_names,
        groundtruth_images,
        groundtruth_images_names,
    )

    # split datasets
    training_images_arr, training_images_count, groundtruth_images_arr, groundtruth_images_count = training_split(
        training_images,
        groundtruth_images,
    )
    (
        training_images_augraphy_arr,
        training_images_augraphy_count,
        groundtruth_images_arr,
        groundtruth_images_count,
    ) = training_split(training_images_augraphy, groundtruth_images)

    # build models
    convolution_kernel_shape = (3, 3)
    callback = EarlyStopping(monitor="loss", patience=30)

    training_instances = generate_training_instances(
        training_images_arr,
        training_images_count,
        training_images_augraphy_arr,
        training_images_augraphy_count,
        groundtruth_images_arr,
        groundtruth_images_count,
        convolution_kernel_shape,
        callback,
    )

    # train models
    for instance in training_instances:
        instance.fit_model()

    # optionally save models
    if save_models:
        for instance in training_instances:
            model_size = instance.model_size
            training_provenance = instance.training_provenance
            loss_function = instance.loss_function
            instance.history.save(f"{model_size}-{training_provenance}-{loss_function}")
