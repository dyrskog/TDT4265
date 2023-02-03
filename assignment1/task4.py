import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
from task3 import SoftmaxTrainer
np.random.seed(0)

def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)
    # You can finish the rest of task 4 below this point.
    weight = model.w
    weight1 = model1.w


    weight = np.transpose(weight[:-1,:]).reshape(10,28,28)
    weight = np.concatenate(weight, axis=1)
    weight1 = np.transpose(weight1[:-1,:]).reshape(10,28,28)
    weight1 = np.concatenate(weight1, axis=1)
    fig, (ax1, ax2) = plt.subplots(2,1)

    ax1.imshow(weight, cmap="gray")
    ax1.set_title(f'lambda = 0')
    ax1.axis('off')
    ax2.imshow(weight1, cmap="gray")
    ax2.set_title(f'lambda = 1')
    ax2.axis('off')

    plt.show()

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]
    
    l2_norms = []
    for l2_lambda in l2_lambdas:
        model = SoftmaxModel(l2_reg_lambda=l2_lambda)
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)
        utils.plot_loss(val_history["accuracy"], f"lambda = {l2_lambda}")
        l2_norms.append(np.linalg.norm(model.w))

    plt.ylim([0.73, 0.93])
    plt.legend()
    plt.show()

    # Task 4d - Plotting of the l2 norm for each weight
    plt.plot(l2_norms)
    plt.xticks(np.arange(len(l2_lambdas)), l2_lambdas)
    plt.show()

if __name__ == "__main__":
    main()