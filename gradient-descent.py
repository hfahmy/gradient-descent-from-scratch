import numpy as np
import pandas as pd
import matplotlib as plt


points = np.array([
    [0, 1.4],
    [2.3, 1.9],
    [2.9, 3.2]
])

# Gradient descent steps

# 0. Define y

def estimate_y(x: np.array, a: float, b: float) -> np.array:
    y = a + x * b
    return y

# 1. Initialize parameters
initial_guess_a, initial_guess_b = 0, 1
learning_rate = 0.01
max_iterations = 1000

# 2. Define the loss functions

def calculate_loss_a(y: np.array, y_hat: np.array) -> float:
    """
    Calculate the loss in y with respect to change in a
    """
    loss =-2 * np.sum(y - y_hat)
    return loss

def calculate_loss_b(y: np.array, y_hat: np.array, x: np.array) -> float:
    """
    Calculate the loss in y with respect to change in b
    """
    loss = -2 * np.sum(x * (y - y_hat))
    return loss

# 3. Calculate the loss at the initial values
y_hat = estimate_y(points[:, 0], initial_guess_a, initial_guess_b)
loss_a = calculate_loss_a(points[:, 1], y_hat)
loss_b = calculate_loss_b(points[:, 1], y_hat, points[:, 0])
