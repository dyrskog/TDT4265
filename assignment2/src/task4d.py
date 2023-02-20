import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


def compare_plot(train_history, val_history, train_history_improved, val_history_improved, description: str):
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Task 2 Model", npoints_to_average=10)
    utils.plot_loss(
        train_history_improved["loss"], description, npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1])
    utils.plot_loss(val_history["accuracy"], "Task 2 Model")
    utils.plot_loss(
        val_history_improved["accuracy"], description)
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()


def compare_all_plots(train_history, val_history, train_histories_improved: list, val_histories_improved:list):
    plt.subplot(1, 2, 1)
    utils.plot_loss(train_history["loss"],
                    "Base Model", npoints_to_average=10)
    for train_history_improved in train_histories_improved:
        utils.plot_loss(
            train_history_improved["loss"], train_history_improved["Description"], npoints_to_average=10)
    plt.ylim([0, .4])
    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 0.98])
    utils.plot_loss(val_history["accuracy"], "Base Model")
    for val_history_improved in val_histories_improved:
        utils.plot_loss(
            val_history_improved["accuracy"], val_history_improved["Description"])
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.show()


# hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
num_epochs = 50
learning_rate = .1
batch_size = 32
momentum_gamma = .9  # Task 3 hyperparameter
shuffle_data = True
use_improved_sigmoid = True
use_improved_weight_init = True
use_momentum = False
use_relu = False

# Load dataset
X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
X_train = pre_process_images(X_train)
X_val = pre_process_images(X_val)
Y_train = one_hot_encode(Y_train, 10)
Y_val = one_hot_encode(Y_val, 10)


# Start with model from task 2
neurons_per_layer = [64, 10]

model = SoftmaxModel(
    neurons_per_layer,
    use_improved_sigmoid,
    use_improved_weight_init,
    use_relu)
trainer = SoftmaxTrainer(
    momentum_gamma, use_momentum,
    model, learning_rate, batch_size, shuffle_data,
    X_train, Y_train, X_val, Y_val,
)
train_history, val_history = trainer.train(num_epochs)


neurons_per_layer = [72, 72, 10]

model = SoftmaxModel(
    neurons_per_layer,
    use_improved_sigmoid,
    use_improved_weight_init,
    use_relu)
trainer = SoftmaxTrainer(
    momentum_gamma, use_momentum,
    model, learning_rate, batch_size, shuffle_data,
    X_train, Y_train, X_val, Y_val,
)
train_history2, val_history2 = trainer.train(num_epochs)
train_history2["Description"] = "2 hidden layers with 72 nodes"
val_history2["Description"] = "2 hidden layers with 72 nodes"

# Task 4e

neurons_per_layer = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 10]

model = SoftmaxModel(
    neurons_per_layer,
    use_improved_sigmoid,
    use_improved_weight_init,
    use_relu)
trainer = SoftmaxTrainer(
    momentum_gamma, use_momentum,
    model, learning_rate, batch_size, shuffle_data,
    X_train, Y_train, X_val, Y_val,
)
train_history3, val_history3 = trainer.train(num_epochs)
train_history3["Description"] = "10 hidden layers with 64 nodes"
val_history3["Description"] = "10 hidden layers with 64 nodes"


train_histories = [train_history2,
                   train_history3]

val_histories = [val_history2,
                 val_history3]

compare_all_plots(train_history, val_history, train_histories, val_histories)

# compare_plot(train_history, val_history, train_history2, val_history2, "2 hidden layers with 72 nodes")


