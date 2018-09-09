import numpy as np


def sigmoid(network, val, derivative=False):  # TODO delete network paramater and fix in NN
    if derivative:
        return val * (1 - val)
    else:
        return 1 / (1 + np.exp(-val))


def relu(network, val, derivative=False):  # TODO delete network paramater and fix in NN
    if derivative:
        return 1 if val > 0 else 0
    else:
        return val if val > 0 else 0


def linear(network, val, derivative=False):
    if derivative:
        return 1
    else:
        return val
