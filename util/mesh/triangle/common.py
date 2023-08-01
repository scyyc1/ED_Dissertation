import numpy as np
from collections import Counter

def retrieve_boundary_edges(faces):
    edges = []
    for triangle in faces:
        for i in range(3):
            p1 = triangle[i]
            p2 = triangle[(i+1)%3]

            edge = (p1, p2) if p1 < p2 else (p2, p1)
            edges.append(edge)
    
    counts = Counter(edges)
    return [item for item, count in counts.items() if count == 1]

def retrieve_all_edges(faces):
    edges = set()

    for triangle in faces:
        for i in range(3):
            p1 = triangle[i]
            p2 = triangle[(i+1)%3]

            edge = (p1, p2) if p1 < p2 else (p2, p1)
            edges.add(edge)
    return list(edges)

def retrieve_boundary_vertices(faces):
    boundary_edges = retrieve_boundary_edges(faces)
    return np.unique(boundary_edges)
    
def retrieve_boundary_angles(boundary_edges, vertices):
    edge_dict = {}
    for edge in boundary_edges:
        for vertex in edge:
            if vertex not in edge_dict:
                edge_dict[vertex] = []
            edge_dict[vertex].append(edge)
    
    boundary_angles = {}
    for vertex, edges in edge_dict.items():
        vectors = []
        for edge in edges:
            if edge[0] != vertex:
                edge = edge[::-1]
            start, end = vertices[edge[0]], vertices[edge[1]]
            vector = end - start
            unit_vector = vector / np.linalg.norm(vector)
            vectors.append(unit_vector)
        cos_angle = np.dot(vectors[0], vectors[1])
        boundary_angles[vertex] = cos_angle
    return boundary_angles