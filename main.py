from time import perf_counter_ns

from keras.callbacks import EarlyStopping

from braun import *


if __name__ == "__main__":

    # we want to save the models we train at the end of this
    save_models = True

    print(f"Save models?: {save_models}")
    # download NoisyOffice data and put it where we need it
    file_setup()
    print("We changed file_setup() to not download NoisyOffice.")
    # print("Downloading NoisyOffice files, building directories.")
    print(
        "We already have NoisyOffice data, and we already cleared the other directories, so we begin.",
    )

    # import the NoisyOffice data
    print("Importing NoisyOffice data")
    (
        training_images_names,
        groundtruth_images_names,
        validation_images_names,
    ) = generate_imageset_names()
    (training_images, groundtruth_images, validation_images) = generate_and_preprocess_imagesets(
        training_images_names,
        groundtruth_images_names,
        validation_images_names,
    )

    # generate and import Augraphy data
    print("Generating and importing Augraphy data")
    start = perf_counter_ns()
    training_images_augraphy = generate_training_images_augraphy()
    training_images_augraphy_names = generate_training_images_augraphy_names()
    end = perf_counter_ns()
    print(
        f"Took {(end - start)/1000000000}s to generate {len(training_images_augraphy_names)} images",
    )

    # display NoisyOffice images
    # plot_training_vs_groundtruth_images(
    #     training_images,
    #     training_images_names,
    #     groundtruth_images,
    #     groundtruth_images_names,
    # )

    # display Augraphy images
    # plot_training_vs_groundtruth_images(
    #     training_images_augraphy,
    #     training_images_augraphy_names,
    #     groundtruth_images,
    #     groundtruth_images_names,
    # )

    # split datasets
    print("Splitting datasets")
    (training_images_arr, training_images_count, groundtruth_images_arr, groundtruth_images_count) = training_split(
        training_images,
        groundtruth_images,
    )

    (
        training_images_augraphy_arr,
        training_images_augraphy_count,
        groundtruth_images_arr,
        groundtruth_images_count,
    ) = training_split(
        training_images_augraphy,
        groundtruth_images,
    )

    # build training instances
    convolution_kernel_shape = (3, 3)
    callback = EarlyStopping(monitor="loss", patience=30)

    print("Generating training instances.")
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
        print(
            f"Training model: {instance.model_size}-{instance.training_provenance}-{instance.loss_function}",
        )
        instance.fit_model()

    # optionally save models
    if save_models:
        for instance in training_instances:
            model_size = instance.model_size
            training_provenance = instance.training_provenance
            loss_function = instance.loss_function
            instance.history.save(f"{model_size}-{training_provenance}-{loss_function}")
