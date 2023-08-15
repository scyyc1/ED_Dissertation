import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from util.mesh.triangle.R2 import retrieve_boundary_angles_2D, retrieve_adjacent_vertices_with_boundary_vertex, retrieve_adjacent_vertices_with_vertex, retrieve_boundary_edges_related_vertices_2D
from util.mesh.triangle.algorithm.R2.embedding_Tutte import Tutte_embedding_2D
from util.mesh.triangle.common import retrieve_boundary_edges, retrieve_boundary_vertices
from util.util import distance_euclidean

class Liu_2017:
    def __init__(self, vertices, faces, lambda_=5, max_iter = 30):
        self.max_iter = max_iter
        self.vertex_num = vertices.shape[0]
        self.vertices = vertices
        self.faces = faces
        self.lambda_=lambda_
        self.loss_history = []
        
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
            
        print(self.lambda_, EB, ED)
            
        return self.lambda_*EB+ED
    
    def optimize(self):
        x0 = np.ravel(self.solution)
        self.res = minimize(self.loss, x0, options = {'maxiter': self.max_iter}, callback=self.callback)
        self.solution = self.res.x.reshape((self.vertex_num, 2))
    
    def optimize_one_round(self):
        x0 = np.ravel(self.solution)
        self.res = minimize(self.loss, x0, options = {'maxiter': 1}, callback=self.callback)
        self.solution = self.res.x.reshape((self.vertex_num, 2))
        
    def callback(self, x0):
        self.loss_history.append(self.objective(x0))
        
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
        
    def v_loss(self, save_dict=''):
        iterations = list(range(1, len(self.loss_history) + 1))
        plt.plot(iterations, self.loss_history, '-o', label='Loss Value', markersize=3)
        plt.title('Loss vs. Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        if save_dict:
            plt.savefig(save_dict, dpi=300)
        plt.legend()
        plt.show()
        
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
        
class Chen_2023_scipy:
    def __init__(self, vertices, faces, lambda1=1, lambda2=1, max_iter = 100):
        self.max_iter = max_iter
        self.v_num = vertices.shape[0]
        self.vertices = vertices
        self.solution = self.vertices.copy()
        self.faces = faces
        self.J = []
        self.b = []
        
        # Hyper parameters
        self.lambda1=lambda1
        self.lambda2=lambda2
        
        # Boundary realated "BE = boundary edges" and "BV = boundary vertices"
        self.BE_r_V = retrieve_boundary_edges(faces)
        self.BV_r_V = retrieve_boundary_vertices(faces)
        self.BV_r_BV = retrieve_adjacent_vertices_with_boundary_vertex(self.BV_r_V, self.BE_r_V)
        self.BE_r_BV = retrieve_boundary_edges_related_vertices_2D(self.BV_r_V, self.BE_r_V)
        self.BV_num = len(self.BV_r_V)
    
    def activation(self, angle):
        if angle < (np.pi/2):
            return np.power(np.cos(angle), 2) / angle
        else:
            return np.power(np.sin(2*angle), 2)
        
    def objective(self, BV):
        BV = BV.reshape((self.BV_num,2))

        E_align = 0
        E_angle = 0
        
        for i, (v1, v2) in enumerate(self.BV_r_BV):
            edge1 = BV[v1] - BV[i]
            edge2 = BV[v2] - BV[i]
            
            L1 = np.linalg.norm(edge1)
            L2 = np.linalg.norm(edge2)
            
            cos_theta = np.dot(edge1, edge2) / (L1 * L2)
            angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            
            E_angle += self.activation(angle)
            E_align += L1*(np.sum(np.absolute(edge1/L1)) - 1) + L2*(np.sum(np.absolute(edge2/L2)) - 1)
        
#         for edge in self.BE_r_BV:
#             A, B = BV[edge[0]], BV[edge[1]]
#             E_align += np.sum(np.absolute(B - A)) - distance_euclidean(A, B)

        return E_angle + E_align
    
    def mapping(self):
        self.solution = Tutte_embedding_2D(self.vertices, self.faces, self.solution[self.BV_r_V])
        
        for face in self.faces:
            A, B, C = self.solution[face]
            A_, B_, C_ = self.vertices[face]
            
            before = np.column_stack([B_-A_,C_-A_])
            after = np.column_stack([B-A,C-A])
            J_ = after @ np.linalg.inv(before)
            b_ = J_ @ B_ - B
            self.J.append(J_)
            self.b.append(b_)
            
    
    def optimize_default(self):
        x0 = np.ravel(self.solution[self.BV_r_V])
        self.res = minimize(self.objective, x0, options = {'maxiter': self.max_iter}, method = "BFGS")
        self.solution[self.BV_r_V] = self.res.x.reshape((len(self.BV_r_V) , 2))
    
    def optimize(self, iter_num):
        for i in range(iter_num):
            self.optimize_one_round()
            print("Round ", i, " done!")
    
    def optimize_one_round(self):
        x0 = np.ravel(self.solution[self.BV_r_V])
        self.res = minimize(self.objective, x0, options = {'maxiter': 1}, method = "BFGS")
        self.solution[self.BV_r_V] = self.res.x.reshape((len(self.BV_r_V) , 2))
        
    def v_plt(self, show_origin=False, show_BV=False, show_vertices=False, show_inner_edges=False, save_dict='', show_boundary_v=False):
#         plt.triplot(self.vertices[:,0], self.vertices[:,1], self.faces, label='Original Mesh', color='blue')
        if show_inner_edges:
            plt.triplot(self.solution[:,0], self.solution[:,1], self.faces, color='skyblue')
        if show_vertices:
            plt.plot(self.vertices[:,0], self.vertices[:,1], 'o')
        if show_BV:
            BV = self.solution[self.BV_r_V]
            plt.plot(BV[:,0], BV[:,1], 'o', color="green")
        if show_origin:
            for edge in self.BE_r_V:
                plt.plot(self.vertices[np.array(edge), 0], self.vertices[np.array(edge), 1], 'g-')
            plt.plot(self.vertices[np.array(self.BE_r_V[0]), 0], self.vertices[np.array(self.BE_r_V[0]), 1],label='Boundary before mapping', color='green')
            if show_inner_edges:
                plt.triplot(self.vertices[:,0], self.vertices[:,1], self.faces, color='greenyellow')
        for edge in self.BE_r_V:
            plt.plot(self.solution[np.array(edge), 0], self.solution[np.array(edge), 1], 'b-')
        plt.plot(self.solution[np.array(self.BE_r_V[0]), 0], self.solution[np.array(self.BE_r_V[0]), 1], label='Boundary after mapping', color='blue')
        if show_boundary_v:
            plt.plot(self.solution[self.BV_r_V,0], self.solution[self.BV_r_V,1], 'o')
        plt.axis('equal')
        plt.legend()
        if save_dict:
            plt.savefig(save_dict, dpi=300)
        plt.show()