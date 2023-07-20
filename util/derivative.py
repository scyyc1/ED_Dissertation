import numpy as np

def derivative_cross_product(A, B, dA, dB):
    return np.cross(dA, B) + np.cross(A, dB)

def derivative_L2_norm(v, dv_dt):
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return 0
    else:
        df_dt = np.dot(v, dv_dt) / v_norm
        return df_dt