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
    
def derivative_L1_norm(v):
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return 0
    
    return np.sign(v)

def derivative_euclidean_distance(p1, p2):
    assert p1.shape == p2.shape
    d = np.linalg.norm(p1 - p2)
    
    dd_dp1 = (p1 - p2) / d
    dd_dp2 = (p2 - p1) / d
    
    return dd_dp1, dd_dp2