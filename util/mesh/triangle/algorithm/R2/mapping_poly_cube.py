import numpy as np
from scipy.optimize import minimize
from util.mesh.triangle.R3 import calculate_single_dihedral_angle, calculate_all_dihedral_angles, calculate_face_normals
from util.mesh.triangle.common import retrieve_all_edges

class chen_2023:
    def __init__(self, vertices, faces, lambda1=1, lambda2=1, max_iter = 30):
        self.max_iter = max_iter
        self.vertex_num = vertices.shape[0]
        self.vertices = vertices
        self.faces = faces
        self.lambda1=lambda1
        self.lambda2=lambda2
        
        self.solution = self.vertices.copy()
        self.edges = retrieve_all_edges(faces)
    
    def activation(self, angle):
        if angle < (np.pi/2):
            return np.power(np.cos(angle), 2) / angle
        else:
            return np.power(np.cos(angle), 2)
        
    def loss_classification(self, x):
        X = x.reshape((self.vertex_num , 3))
        
        EB = 0
        for i, face in enumerate(self.faces):
            v1, v2, v3 = X[face[0]], X[face[1]], X[face[2]]
            a = v2 - v1
            b = v3 - v1
            cross_product = np.cross(a, b)
#             normal = cross_product / np.linalg.norm(cross_product)
            area = np.linalg.norm(cross_product) / 2
#             EB += area * (np.abs(normal) - 1)
            EB += np.sum(np.abs(cross_product)) - area
#         print(EB)
            
        EA = 0
        dihedral_angles = calculate_all_dihedral_angles(faces, X)
        for angle_value in dihedral_angles.values():
            EA += self.activation(angle_value)
            
        return self.lambda1*EB + self.lambda2*EA
    
    def optimize(self):
        x0 = np.ravel(self.solution)
        self.res = minimize(self.loss_classification, x0, options = {'maxiter': self.max_iter})
        self.solution = self.res.x.reshape((self.vertex_num, 3))
    
    def optimize_one_round(self):
        x0 = np.ravel(self.solution)
        self.res = minimize(self.loss_classification, x0, options = {'maxiter': 1})
        self.solution = self.res.x.reshape((self.vertex_num, 3))
    