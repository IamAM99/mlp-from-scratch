"""Necessary functions for MLP implementation.
"""

import numpy as np
import matplotlib.pyplot as plt


def shuffle(a: np.ndarray, b: np.ndarray):
    """Shuffles a given pair of numpy arrays based on their first axis.
    """
    idx = np.random.permutation(a.shape[0])
    return a[idx], b[idx]


def add_bias(matrix: np.ndarray):
    """Adds bias to the first column of matrix.
    """
    matrix = np.atleast_2d(matrix)
    return np.c_[np.ones(matrix.shape[0]), matrix]


def hp_tuning(
    X: np.ndarray,
    y: np.ndarray,
    MLP: "MLP",
    neurons: list,
    learning_rate: list,
    iterations: int,
    n_fit: int = 5,
):
    """A function to help with hyper-parameter tuning
    """
    plt.figure()
    legend = []
    for neurons_ in neurons:
        for learning_rate_ in learning_rate:
            hist = np.zeros(iterations)
            for _ in range(n_fit):
                # Create the network
                mlp = MLP(n_features=1, hidden_neurons=neurons_)

                # Train the network
                hist += mlp.fit(
                    X, y, iterations=iterations, learning_rate=learning_rate_
                )
            hist /= n_fit

            # Plot the error
            plt.plot(np.log10(hist))
            legend.append(f"n={neurons_}, lr={learning_rate_}")
    plt.legend(legend)
    plt.title("Mean Square Error per Iteration")
    plt.xlabel("iteration")
    plt.ylabel("MSE (log scale)")
    plt.show()
