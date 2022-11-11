import numpy as np

def pearson_correlation_coefficient(y_hat, y):
    return np.corrcoef(y_hat, y)[0, 1]