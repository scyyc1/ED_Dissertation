import numpy as np

def vector_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def calculate_normal_2D(A, B):
    AB = B - A
    normal = np.array([-AB[1], AB[0]])
    return vector_normalize(normal)

def calculate_L1_norm(v):
    return np.sum(np.abs(v))