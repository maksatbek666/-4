import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def chebyshev_distance(x1, x2):
    return np.max(np.abs(x1 - x2))



print ("евклидово расстояние",euclidean_distance( 50,45))
print("манхэттенское расстояние",manhattan_distance(50,45))
print("chebyshev_distance",chebyshev_distance(50,45))