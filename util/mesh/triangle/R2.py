from util.triangle import calculate_signed_area
import numpy as np

def make_clockwise_2D(vertices, faces):
    for i in range(len(faces)):
        A, B, C = vertices[faces[i]]
        if calculate_signed_area(A, B, C) > 0:
            faces[i] = faces[i][::-1]
    return faces

def retrieve_boundary_angles_2D(boundary_edges, vertices):
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
        boundary_angles[vertex] = np.arccos(cos_angle)
    return boundary_angles