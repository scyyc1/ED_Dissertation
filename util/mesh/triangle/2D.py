from util.triangle import calculate_signed_area

def make_clockwise_2D(vertices, faces):
    for i in range(len(faces)):
        A, B, C = vertices[faces[i]]
        if calculate_signed_area(A, B, C) > 0:
            faces[i] = faces[i][::-1]
    return faces