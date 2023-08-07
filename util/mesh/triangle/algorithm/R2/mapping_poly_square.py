import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from util.mesh.triangle.R2 import retrieve_boundary_angles_2D
from util.mesh.triangle.common import retrieve_boundary_edges
from util.util import distance_euclidean

class Liu_2017:
    def __init__(self, vertices, faces, lambda_=5, max_iter = 30):
        self.max_iter = max_iter
        self.vertex_num = vertices.shape[0]
        self.vertices = vertices
        self.faces = faces
        self.lambda_=lambda_
        
        self.solution = self.vertices.copy()
        self.boundary_edges = retrieve_boundary_edges(faces)
        
    def loss(self, x):
        X = x.reshape((self.vertex_num, 2))
        
        # Boundary edges alignment
        EB = 0
        for edge in self.boundary_edges:
            A, B = X[edge[0]], X[edge[1]]
            EB += np.sum(np.absolute(B - A)) - distance_euclidean(A, B)
            
        # Distortion term
        ED = 0
        for face in self.faces:
            A, B, C = X[face[0]], X[face[1]], X[face[2]]
            A_, B_, C_ = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]

            before = np.column_stack([B_-A_,C_-A_])
            after = np.column_stack([B-A,C-A])
            J = after @ np.linalg.inv(before)

            u, s, v = np.linalg.svd(J)
            s1, s2 = s[0], s[1]
            ED += np.exp(s1/s2 + s2/s1)
            
        return self.lambda_*EB+ED
    
    def optimize(self):
        x0 = np.ravel(self.solution)
        self.res = minimize(self.loss, x0, options = {'maxiter': self.max_iter})
        self.solution = self.res.x.reshape((self.vertex_num, 2))
    
    def optimize_one_round(self):
        x0 = np.ravel(self.solution)
        self.res = minimize(self.loss, x0, options = {'maxiter': 1})
        self.solution = self.res.x.reshape((self.vertex_num, 2))
        
    def visualize_initial(self, show_boundary = False):
        plt.plot(self.vertices[:,0], self.vertices[:,1], 'o', color='green')
        plt.triplot(self.vertices[:,0], self.vertices[:,1], self.faces, label='Original Mesh', color='orange')
        if show_boundary:
            for edge in self.boundary_edges:
                plt.plot(self.vertices[np.array(edge), 0], self.vertices[np.array(edge), 1], 'y-')
        plt.axis('equal')
        plt.legend()
#         plt.show()
        
    def visualize_solution(self, show_boundary = False):
        plt.plot(self.solution[:,0], self.solution[:,1], 'o', color='blue')
        plt.triplot(self.solution[:,0], self.solution[:,1], self.faces, label='Transformed Mesh', color='blue')
        if show_boundary:
            for edge in self.boundary_edges:
                plt.plot(self.solution[np.array(edge), 0], self.solution[np.array(edge), 1], 'r-')
        plt.axis('equal')
        plt.legend()
#         plt.show()
        
    def visualize(self):
        self.visualize_initial()
        self.visualize_solution()
        
class Chen_2023_ver1:
    def __init__(self, vertices, faces, lambda1=1, lambda2=1, max_iter = 30):
        self.max_iter = max_iter
        self.vertex_num = vertices.shape[0]
        self.vertices = vertices
        self.faces = faces
        self.lambda1=lambda1
        self.lambda2=lambda2
        
        self.solution = self.vertices.copy()
        self.boundary_edges = retrieve_boundary_edges(faces)
    
    def activation(self, angle):
        if angle < (np.pi/2):
            return np.power(np.cos(angle), 2) / angle
        else:
            return np.power(np.cos(angle), 2)
        
    def loss_classification(self, x):
        X = x.reshape((self.vertex_num , 2))
        
        EB = 0
        for edge in self.boundary_edges:
            A, B = X[edge[0]], X[edge[1]]
            EB += np.sum(np.absolute(B - A)) - distance_euclidean(A, B)
            
        EA = 0
        boundary_angles = retrieve_boundary_angles_2D(self.boundary_edges, X)
        for angle_value in boundary_angles.values():
            EA += self.activation(angle_value)
            
        return self.lambda1*EB + self.lambda2*EA
    
    def optimize_fault(self):
        x0 = np.ravel(self.solution)
        self.res = minimize(self.loss_classification, x0, options = {'maxiter': self.max_iter})
        self.solution = self.res.x.reshape((self.vertex_num, 2))
    
    def optimize(self, iter_num):
        for i in range(iter_num):
            self.optimize_one_round()
            print("Round ", i, " done!")
    
    def optimize_one_round(self):
        x0 = np.ravel(self.solution)
        self.res = minimize(self.loss_classification, x0, options = {'maxiter': 1})
        self.solution = self.res.x.reshape((self.vertex_num, 2))
        
    def visualize_initial(self, show_boundary = False):
        plt.plot(self.vertices[:,0], self.vertices[:,1], 'o', color='green')
        plt.triplot(self.vertices[:,0], self.vertices[:,1], self.faces, label='Original Mesh', color='orange')
        if show_boundary:
            for edge in self.boundary_edges:
                plt.plot(self.vertices[np.array(edge), 0], self.vertices[np.array(edge), 1], 'y-')
        plt.axis('equal')
        plt.legend()
        
    def visualize_solution(self, show_boundary = False):
        plt.plot(self.solution[:,0], self.solution[:,1], 'o', color='blue')
        plt.triplot(self.solution[:,0], self.solution[:,1], self.faces, label='Transformed Mesh', color='blue')
        if show_boundary:
            for edge in self.boundary_edges:
                plt.plot(self.solution[np.array(edge), 0], self.solution[np.array(edge), 1], 'r-')
        plt.axis('equal')
        plt.legend()
        
    def visualize(self):
        self.visualize_initial()
        self.visualize_solution()