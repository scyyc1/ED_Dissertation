import numpy as np
import pyvista as pv

def convert_to_pyvista_faces(faces):
    pyvista_faces = np.hstack((3 * np.ones((faces.shape[0], 1), dtype=int), faces))
    return pyvista_faces.flatten()

def preprocess(path, scale=1):
    mesh = pv.read(path)
    vertices = mesh.points[:, :-1].copy()*scale
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    return vertices, faces

def postprocess(path, vertices, faces):
    pyvista_vertices = np.hstack((vertices, np.zeros((len(vertices), 1))))
    pyvista_faces = convert_to_pyvista_faces(faces)
    
    output = pv.PolyData(pyvista_vertices, pyvista_faces)
    output.save(path)