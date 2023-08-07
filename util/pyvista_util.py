import numpy as np

def convert_to_pyvista_faces(faces):
    pyvista_faces = np.hstack((3 * np.ones((faces.shape[0], 1), dtype=int), faces))
    return pyvista_faces.flatten()