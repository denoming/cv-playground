import numpy as np
import matplotlib.pyplot as plt


def plot_image(image: any):
    fig = plt.figure()
    ax1 = fig.add_axes((0, 0, 1, 1))
    ax1.imshow(image)
    plt.show()


def plot_vectors(list_v, list_label, list_color):
    _, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-10, 10))
    ax.set_yticks(np.arange(-10, 10))

    plt.axis((-10, 10, -10, 10))
    for i, v in enumerate(list_v):
        sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(v)])
        plt.quiver(v[0], v[1], color=list_color[i], angles='xy', scale_units='xy', scale=1)
        ax.text(v[0] - 0.2 + sgn[0], v[1] - 0.2 + sgn[1], list_label[i], fontsize=14, color=list_color[i])

    plt.grid()
    plt.gca().set_aspect("equal")
    plt.show()


def plot_transformation(T, e1, e2):
    color_original = "#129cab"
    color_transformed = "#cc8933"

    _, ax = plt.subplots(figsize=(7, 7))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-5, 5))
    ax.set_yticks(np.arange(-5, 5))

    plt.axis((-5, 5, -5, 5))
    plt.quiver([0, 0], [0, 0], [e1[0], e2[0]], [e1[1], e2[1]], color=color_original, angles='xy', scale_units='xy',
               scale=1)
    plt.plot([0, e2[0], e1[0], e1[0]],
             [0, e2[1], e2[1], e1[1]],
             color=color_original)
    e1_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(e1)])
    ax.text(e1[0] - 0.2 + e1_sgn[0], e1[1] - 0.2 + e1_sgn[1], f'$e_1$', fontsize=14, color=color_original)
    e2_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(e2)])
    ax.text(e2[0] - 0.2 + e2_sgn[0], e2[1] - 0.2 + e2_sgn[1], f'$e_2$', fontsize=14, color=color_original)

    e1_transformed = T(e1)
    e2_transformed = T(e2)

    plt.quiver([0, 0], [0, 0], [e1_transformed[0], e2_transformed[0]], [e1_transformed[1], e2_transformed[1]],
               color=color_transformed, angles='xy', scale_units='xy', scale=1)
    plt.plot([0, e2_transformed[0], e1_transformed[0] + e2_transformed[0], e1_transformed[0]],
             [0, e2_transformed[1], e1_transformed[1] + e2_transformed[1], e1_transformed[1]],
             color=color_transformed)
    e1_transformed_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(e1_transformed)])
    ax.text(e1_transformed[0][0] - 0.2 + e1_transformed_sgn[0], e1_transformed[1][0] - e1_transformed_sgn[1][0],
            f'$T(e_1)$', fontsize=14, color=color_transformed)
    e2_transformed_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(e2_transformed)])
    ax.text(e2_transformed[0][0] - 0.2 + e2_transformed_sgn[0][0], e2_transformed[1][0] - e2_transformed_sgn[1][0],
            f'$T(e_2)$', fontsize=14, color=color_transformed)

    plt.gca().set_aspect("equal")
    plt.show()


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions=None):
    """
    Plots linear training data and test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    plt.legend(prop={"size": 14})


def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()