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

def retrieve_triangles_with_edge(faces, edges):
    edge_to_triangles = {edge: [] for edge in edges}

    for triangle in faces:
        for i in range(3):
            p1 = triangle[i]
            p2 = triangle[(i+1)%3]

            edge = (p1, p2) if p1 < p2 else (p2, p1)
            edge_to_triangles[edge].append(triangle)

    return edge_to_triangles