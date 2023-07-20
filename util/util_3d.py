import numpy as np
import random

def line_plane_intersection(p, n, plane):
    a, b, c, d = plane
    x0, y0, z0 = p
    n_x, n_y, n_z = n

    t = - (a*x0 + b*y0 + c*z0 + d) / (a*n_x + b*n_y + c*n_z)

    x = x0 + t*n_x
    y = y0 + t*n_y
    z = z0 + t*n_z

    return np.array([x, y, z])

class line_3D:
    def __init__(self, p, n):
        norm = np.linalg.norm(n)
        if not norm==1:
            n = vector_normalize(n)
        self.p=p
        self.n=n
        
    def get_point(self, t):
        return self.p + self.n * t
    
    def intersection_plane_t(self, plane):
        A, B, C, D = plane
        N = np.array([A, B, C])
        return -(np.dot(N, self.p) + D) / np.dot(N, self.n)
    
    def intersection_plane_point(self, plane):
        t = self.intersection_plane_t(plane)
        return self.get_point(t)


def vector_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def make_clockwise_3d(triangle):
    vec1 = triangle[1] - triangle[0]
    vec2 = triangle[2] - triangle[0]
    normal = np.cross(vec1, vec2)

    if normal[2] < 0:
        return triangle[::-1]
    else:
        return triangle