import numpy as np

def calculate_area(A, B, C):
    return np.linalg.norm(np.cross(B-A, C-A)) / 2

def calculate_signed_area(A, B, C):
    return np.cross(B-A, C-A) / 2