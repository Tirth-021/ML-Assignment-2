import numpy as np

def generate_dataset1():
    num_samples = 40
    np.random.seed(45)
    x1 = np.random.uniform(-20, 20, num_samples)
    f_x = 100 * x1 + 1
    eps = np.random.randn(num_samples)
    Y = f_x + eps
    X = np.c_[np.ones(num_samples), x1]  # add intercept term
    return X, Y

def generate_dataset2():
    num_samples = 40
    np.random.seed(45)
    x1 = np.random.uniform(-1, 1, num_samples)
    f_x = 3 * x1 + 4
    eps = np.random.randn(num_samples)
    Y = f_x + eps
    X = np.c_[np.ones(num_samples), x1]  # add intercept term
    return X, Y
