import numpy as np
from util.mesh.triangle.common import retrieve_triangles_with_edge, retrieve_all_edges

def make_clockwise_3d(triangle):
    vec1 = triangle[1] - triangle[0]
    vec2 = triangle[2] - triangle[0]
    normal = np.cross(vec1, vec2)

    if normal[2] < 0:
        return triangle[::-1]
    else:
        return triangle
    
def calculate_single_dihedral_angle(A_common, B_common, C, D):
    AB = B_common - A_common
    AC = C - A_common
    AD = D - A_common

    N1 = np.cross(AB, AC) 
    N2 = np.cross(AB, AD) 

    dot_product = np.dot(N1, N2)
    cross_product = np.cross(N1, N2)

    cos_theta = np.clip(dot_product / (np.linalg.norm(N1) * np.linalg.norm(N2)), -1, 1)
    theta = np.arccos(cos_theta)

    if cross_product[2] < 0:
        theta = 2 * np.pi - theta

    return theta

def calculate_all_dihedral_angles(faces, vertices):
    edges = retrieve_all_edges(faces)
    edge_to_triangles = retrieve_triangles_with_edge(faces, edges)
    edge_to_angle = {}
    
    for edge, triangles in edge_to_triangles.items():
        if len(triangles) != 2: 
            continue

        triangle1, triangle2 = triangles

        C = list(set(triangle1) - set(edge))[0]
        D = list(set(triangle2) - set(edge))[0]

        A = vertices[edge[0]]
        B = vertices[edge[1]]
        C = vertices[C]
        D = vertices[D]

        angle = calculate_single_dihedral_angle(A, B, C, D)
        edge_to_angle[edge] = angle

    return edge_to_angle