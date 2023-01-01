from typing import Callable
from typing import List

from tensorflow.keras.models import Model

from braun.model import convnet_denoiser


def build_SML_models(convolution_kernel_shape):
    small = ConvNetModelInstance(
        convolution_kernel_shape,
        "small",
    )
    medium = ConvNetModelInstance(
        convolution_kernel_shape,
        "medium",
    )
    large = ConvNetModelInstance(
        convolution_kernel_shape,
        "large",
    )
    return small, medium, large


def generate_training_instances(
    training_images_arr,
    training_images_count,
    training_images_augraphy_arr,
    training_images_augraphy_count,
    groundtruth_images_arr,
    groundtruth_images_count,
    convolution_kernel_shape,
    callback,
):
    training_instances = []
    small, medium, large = build_SML_models(convolution_kernel_shape)

    for model_type in [small, medium, large]:
        # augraphy instance
        training_instances.append(
            TrainingInstance(
                model_type,
                "augraphy",
                callback,
                training_images_augraphy_arr,
                groundtruth_images_arr,
                training_images_augraphy_count,
                groundtruth_images_count,
                600,
                24,
            ),
        )

        # noisyoffice instance
        training_instances.append(
            TrainingInstance(
                model_type,
                "noisyoffice",
                callback,
                training_images_arr,
                groundtruth_images_arr,
                training_images_count,
                groundtruth_images_count,
                600,
                24,
            ),
        )

    return training_instances


class ConvNetModelInstance:
    def __init__(
        self,
        convolution_kernel_shape,
        model_size,
    ):
        self.model_size = model_size
        self.model = convnet_denoiser(
            convolution_kernel_shape,
            model_size,
            loss_function,
        )


class TrainingInstance(ConvNetModelInstance):
    def __init__(
        self,
        model_instance: ConvNetModelInstance,
        training_provenance: str,
        callback: Callable,
        training_images: List,
        groundtruth_images: List,
        training_images_count: int,
        groundtruth_images_count: int,
        epochs: int,
        batch_size: int,
    ):
        self.model = model_instance.model
        self.training_provenance = training_provenance
        self.model_size = model_instance.model_size
        self.training_images = training_images
        self.groundtruth_images = groundtruth_images
        self.training_images_count = training_images_count
        self.groundtruth_images_count = groundtruth_images_count
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback = callback
        self.history = None

    def fit_model(self):
        self.history = self.model.fit(
            self.training_images,
            self.groundtruth_images,
            validation_data=(self.training_images_count, self.groundtruth_images_count),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=2,
            callbacks=[self.callback],
        )
