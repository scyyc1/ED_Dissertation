import numpy as np
import matplotlib as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from util.mesh.triangle.common import retrieve_boundary_vertices, retrieve_boundary_edges

class Tutte_1963_2D:
    def __init__(self, vertices, faces, name="default"):
        self.num_vertices = len(vertices)
        self.vertices = vertices
        self.faces = faces
        self.name = name
        
        self.boundary_edges = retrieve_boundary_edges(faces)
        self.boundary_vertices = retrieve_boundary_vertices(faces)
        self.W = weights_Tutte(vertices, faces, self.boundary_vertices)

    def mapping(self, boundary_vertices_coord_updated): 
        b = np.zeros((self.num_vertices,2))
        b[self.boundary_vertices] = boundary_vertices_coord_updated
        
        self.result = spsolve(csr_matrix(self.W), b)
        return self.result
    
    def v_plt(self, show_vertices=False, show_boundary=False, show_origin=False, save_dict=''):
        plt.triplot(self.result[:,0], self.result[:,1], self.faces, label='Transformed Mesh', color='black')
        if show_origin:
            plt.triplot(self.vertices[:,0], self.vertices[:,1], self.faces, label='Original Mesh', color='cyan')
        if show_vertices:
            plt.plot(self.result[:,0], self.result[:,1], 'o', color='red')
        if show_boundary:
            for edge in self.boundary_edges:
                plt.plot(self.result[np.array(edge), 0], self.result[np.array(edge), 1], 'r-')
        if save_dict:
            plt.savefig(save_dict, dpi=300)
        plt.axis('equal')
        plt.legend()
        plt.show()
        
        
def Tutte_embedding_2D(vertices, faces, updated_boundary_vertices):
    boundary_vertices = retrieve_boundary_vertices(faces)
    
    W = weights_Tutte(vertices, faces, boundary_vertices)
    
    b = np.zeros((len(vertices),2))
    b[boundary_vertices] = updated_boundary_vertices
        
    return spsolve(csr_matrix(W), b)
    

def weights_Tutte(vertices, faces, boundary_vertices):
    N = len(vertices)
    W = np.zeros((N, N))

    for i, j, k in faces:
        for u, v in [(i, j), (j, k), (k, i)]:
            if u not in boundary_vertices:
                W[u, v] = 1
            if v not in boundary_vertices:
                W[v, u] = 1

    non_boundary_vertices = [i for i in range(N) if i not in boundary_vertices]
    W[non_boundary_vertices] = - W[non_boundary_vertices] / np.sum(W[non_boundary_vertices], axis=1, keepdims=True)
    W += np.eye(N)
    return W