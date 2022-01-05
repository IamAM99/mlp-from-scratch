"""Multi Layer Perceptron class.
"""

import numpy as np
from activations import sigmoid_act, d_sigmoid_act, linear_act, d_linear_act
from utils import shuffle, add_bias


class MLP:
    """Multi-Layer Perceptron class.
    """

    def __init__(self, n_features: int, hidden_neurons: int):
        """Initialize an MLP architecture with one hidden layer, and one output
        neuron.

        Parameters
        ----------
        n_features:
            Number of features, or in other words, number of input dimentions.
        hidden_neurons:
            Number of neurons in the hidden layer.
        """
        # number of neurons in each layer
        n0 = n_features  # input neurons
        n1 = hidden_neurons  # hidden layer neurons
        n2 = 1  # output neurons

        # weights
        self.w1 = np.random.rand(n1, n0 + 1)  # +1 is for the bias neuron
        self.w2 = np.random.rand(n2, n1 + 1)

        # error values for each iteration will be stored in self.error
        self.error = []

    @staticmethod
    def feedforward(x: np.ndarray, w1: np.ndarray, w2: np.ndarray):
        """Calculates the feedforward outputs for each layer.

        Parameters
        ----------
        x:
            Input data sample.
        w1:
            Weights between input and the hidden layer.
        w2:
            Weights between the hidden layer and the output.

        Returns
        -------
        hidden_out:
            Output of the hidden layer.
        d_hidden_out:
            Derivative of the output of the hidden layer.
        output:
            Output of the network.
        d_output:
            derivative of the output of the network.
        """
        # hidden layer
        hidden_net = np.dot(x, w1.T)
        hidden_out = sigmoid_act(hidden_net)
        d_hidden_out = d_sigmoid_act(hidden_net)

        # output layer
        hidden_out = add_bias(hidden_out)
        output_net = np.dot(hidden_out, w2.T)
        output = linear_act(output_net)
        d_output = d_linear_act(output_net)

        return hidden_out, d_hidden_out, output, d_output

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        iterations: int,
        learning_rate: float,
        return_error: bool = True,
    ):
        """Fits the (inputs, targets) dataset onto the model and updates the
        weights.

        Parameters
        ----------
        inputs:
            Dataset of input samples as a numpy array.
        targets:
            Dataset of targets as a numpy array.
        iterations:
            Number of times to iterate over all data samples.
        learning_rate:
            The learning rate for updating the weights.
        return_error:
            Whether to return the list of error values in each iteration or not,
            by default True.

        Returns
        -------
        error: (np.ndarray)
            An array of error values in each iteration; it'll only be returned
            if return_error is True.
        """
        for _ in range(iterations):
            # randomize the dataset in each iteration
            inputs, targets = shuffle(inputs, targets)

            # initialize weight update values in each iteration
            delta_w1 = 0
            delta_w2 = 0

            # initialize a new error for this iteration
            self.error.append(0)
            for x, y in zip(inputs, targets):
                # x.shape = (i,). It should be: (1, i). It is also true for y.shape.
                x = np.atleast_2d(x)
                y = np.atleast_2d(y)

                # add a bias value to the input features
                x = add_bias(x)

                # calculate the feedforward
                y1, d_y1, out, d_out = self.feedforward(x, self.w1, self.w2)

                # calculate the delta_o parameter, used for backpropagation
                delta_o = (y - out) * d_out

                # calculate the weight update values using backpropagation algorithm
                # bias doesn't propagate => we use w2[:, 1:] in the formula
                delta_w1 += delta_o * np.dot((self.w2[:, 1:] * d_y1).T, x)
                delta_w2 += delta_o * y1

                # update the weights
                self.w1 += learning_rate * delta_w1
                self.w2 += learning_rate * delta_w2

                # add up the error for this sample
                self.error[-1] += 0.5 * (y - out).item() ** 2

            # calculate the average of errors over all samples
            self.error[-1] /= inputs.shape[0]
        if return_error:
            return np.array(self.error)
        return None

    def predict(self, X_test: np.ndarray):
        """Calculate the target predictions for a given dataset.

        Parameters
        ----------
        X_test:
            Input dataset.

        Returns
        -------
        y: (np.ndarray)
            An array of predicted target values for the dataset using the network.
        """
        # preprocess input
        X = add_bias(np.atleast_2d(X_test).T)

        # calculate the output
        _, _, y, _ = self.feedforward(X, self.w1, self.w2)
        return y
