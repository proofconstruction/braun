from typing import Tuple
from typing import Union

import tensorflow as tf
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.models import Model
from tensorflow import keras


def encode(
    input_tensor,
    convolution_kernel_shape=Tuple[int, int],
    model_size: str = Union["small", "medium", "large"],
):
    c1 = Conv2D(16, convolution_kernel_shape, activation="relu", padding="same")(input_tensor)
    c2 = Conv2D(16, convolution_kernel_shape, activation="relu", padding="same")(c1)
    c3 = Conv2D(32, convolution_kernel_shape, activation="relu", padding="same")(c2)
    c4 = Conv2D(32, convolution_kernel_shape, activation="relu", padding="same")(c3)
    c5 = Conv2D(64, convolution_kernel_shape, activation="relu", padding="same")(c4)
    c6 = Conv2D(64, convolution_kernel_shape, activation="relu", padding="same")(c5)
    c7 = Conv2D(128, convolution_kernel_shape, activation="relu", padding="same")(c6)
    c8 = Conv2D(128, convolution_kernel_shape, activation="relu", padding="same")(c7)

    if model_size == "small":
        return input_tensor.add(c2)
    elif model_size == "medium":
        return input_tensor.add(c4)
    else:
        return input_tensor.add(c8)


def downsample_dropout(input_tensor, dropout_rate: float = 0.5):
    downsample = MaxPooling2D((2, 2), padding="same")
    dropout = Dropout(dropout_rate)(downsample)
    return input_tensor.add(dropout)


def decode(
    input_tensor,
    convolution_kernel_shape=Tuple[int, int],
    model_size: str = Union["small", "medium", "large"],
):
    c1 = Conv2D(128, convolution_kernel_shape, activation="relu", padding="same")(input_tensor)
    c2 = Conv2D(128, convolution_kernel_shape, activation="relu", padding="same")(c1)
    c3 = Conv2D(64, convolution_kernel_shape, activation="relu", padding="same")(c2)
    c4 = Conv2D(64, convolution_kernel_shape, activation="relu", padding="same")(c3)
    c5 = Conv2D(32, convolution_kernel_shape, activation="relu", padding="same")(c4)
    c6 = Conv2D(32, convolution_kernel_shape, activation="relu", padding="same")(c5)
    c7 = Conv2D(16, convolution_kernel_shape, activation="relu", padding="same")(c6)
    c8 = Conv2D(16, convolution_kernel_shape, activation="relu", padding="same")(c7)

    if model_size == "small":
        return input_tensor.add(c2)
    elif model_size == "medium":
        return input_tensor.add(c4)
    else:
        return input_tensor.add(c8)


def upsample(input_tensor):
    return input_tensor.add(UpSampling2D((2, 2)))


def convnet_denoiser(
    convolution_kernel_shape=Tuple[int, int],
    model_size: str = Union["small", "medium", "large"],
    loss_function: str = Union["cosine_similarity", "mean_absolute_error", "mean_squared_error"],
):
    # build and compile the model
    # I wish we had a nice monad here
    model = tf.keras.Input(shape=(540, 420, 1))
    model = encode(model, convolution_kernel_shape, model_size)
    model = downsample_dropout(model)
    model = decode(model, convolution_kernel_shape, model_size)
    model = upsample(model)
    model = model.add(Conv2D(1, (3, 3), activation="sigmoid", padding="same"))

    return model.compile(optimizer="adam", loss_function=loss_function, metrics=loss_function)
