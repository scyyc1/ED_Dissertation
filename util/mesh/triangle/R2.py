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

def retrieve_boundary_vertices_related_edges_2D(boundary_vertices, boundary_edges):
    vertex_to_edge = {v: [] for v in boundary_vertices}

    for idx, (v1, v2) in enumerate(boundary_edges):
        if v1 in vertex_to_edge:
            vertex_to_edge[v1].append(idx)
        if v2 in vertex_to_edge:
            vertex_to_edge[v2].append(idx)

    related_edge_indices = [vertex_to_edge[v] for v in boundary_vertices]

    return np.array(related_edge_indices)

def retrieve_boundary_edges_related_vertices_2D(boundary_vertices, boundary_edges):
    BE_r_BV = []
    
    for v1, v2 in boundary_edges:
        idx1 = np.where(boundary_vertices==v1)
        idx2 = np.where(boundary_vertices==v2)
        BE_r_BV.append(np.hstack((idx1[0], idx2[0])))
    
    return np.array(BE_r_BV)