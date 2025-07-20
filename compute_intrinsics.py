import numpy as np

def v_ij(H, i, j):
    """
    Helper for forming the V matrix (as in Zhang's method).
    """
    return np.array([
        H[0,i]*H[0,j],
        H[0,i]*H[1,j] + H[1,i]*H[0,j],
        H[1,i]*H[1,j],
        H[2,i]*H[0,j] + H[0,i]*H[2,j],
        H[2,i]*H[1,j] + H[1,i]*H[2,j],
        H[2,i]*H[2,j]
    ])

def compute_intrinsics(H_all):
    """
    Computes the intrinsic camera matrix K from all estimated homographies.
    Args:
        H_all (list of np.ndarray): List of 3x3 homography matrices.
    Returns:
        K (np.ndarray): 3x3 intrinsic camera matrix.
    """
    V = []
    for H in H_all:
        H = H / H[2,2]  # Normalize H just in case
        V.append(v_ij(H, 0, 1))                # v_01
        V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))# v_00 - v_11
    V = np.array(V)
    # Solve Vb = 0 via SVD
    U, S, Vt = np.linalg.svd(V)
    b = Vt[-1,:]

    # Interpret b to fill in the B matrix (Zhang's notation)
    B11, B12, B22, B13, B23, B33 = b
    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lambda_ = B33 - (B13**2 + v0*(B12*B13 - B11*B23))/B11
    alpha = np.sqrt(lambda_ / B11)
    beta = np.sqrt(lambda_ * B11 / (B11*B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lambda_
    u0 = gamma * v0 / beta - B13 * alpha**2 / lambda_
    K = np.array([
        [alpha, gamma, u0],
        [0,     beta,  v0],
        [0,     0,     1]
    ])
    return K