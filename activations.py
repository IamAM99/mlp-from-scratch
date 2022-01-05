"""Activation functions and their derivatives for the MLP implementation.
"""

import numpy as np


def sigmoid_act(x: np.ndarray):
    """Calculates the sigmoid activation function.
    """
    return 1 / (1 + np.exp(-x))


def d_sigmoid_act(x: np.ndarray):
    """Calculates the derivative of the sigmoid activation function.
    """
    return sigmoid_act(x) * (1 - sigmoid_act(x))


def linear_act(x: np.ndarray):
    """Calculates the linear activation function.
    """
    return x


def d_linear_act(x: np.ndarray):
    """Calculates the derivative of the linear activation function.
    """
    return np.atleast_2d(1)
