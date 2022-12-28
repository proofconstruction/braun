import matplotlib.pyplot as plt

from training import TrainingInstance


def plot_training_vs_groundtruth_images(
    training_images,
    training_images_names,
    groundtruth_images,
    groundtruth_images_names,
):
    plt.figure(figsize=(15, 25))
    for i in range(0, 8, 2):
        plt.subplot(4, 2, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(training_images[i][:, :, 0], cmap="gray")
        plt.title(f"Noisy image: {training_images_names[i]}")

        plt.subplot(4, 2, i + 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(groundtruth_images[i][:, :, 0], cmap="gray")
        plt.title(f"Clean image: {groundtruth_images_names[i]}")

    plt.show()


def plot_traininginstance_loss_and_error(training_instance: TrainingInstance):
    history = training_instance.history
    error_type = training_instance.loss_function
    # Check how loss & mae went down
    epoch_loss = history.history["loss"]
    epoch_val_loss = history.history["val_loss"]
    epoch_error = history.history[error_type]
    epoch_val_error = history.history["val_" + error_type]

    plt.figure(figsize=(20, 6), dpi=600)
    plt.subplot(1, 2, 1)
    plt.plot(range(0, len(epoch_loss)), epoch_loss, "b-", linewidth=2, label="Train Loss")
    plt.plot(range(0, len(epoch_val_loss)), epoch_val_loss, "r-", linewidth=2, label="Validation Loss")
    plt.title("Evolution of loss on train & validation datasets over epochs")
    plt.legend(loc="best")

    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(epoch_error)), epoch_error, "b-", linewidth=2, label=f"Train {error_type.upper()}")
    plt.plot(
        range(0, len(epoch_val_error)),
        epoch_val_error,
        "r-",
        linewidth=2,
        label=f"Validation {error_type.upper()}",
    )
    plt.title(f"Evolution of {error_type} on train & validation datasets over epochs")
    plt.legend(loc="best")

    plt.show()
