import numpy as np

def compute_extrinsic(K, H):
    """
    Computes rotation (R) and translation (t) for one view from K and H.

    Args:
        K (np.ndarray): 3x3 camera intrinsic matrix.
        H (np.ndarray): 3x3 homography matrix for that view.

    Returns:
        R (np.ndarray): 3x3 rotation matrix.
        t (np.ndarray): 3x1 translation vector.
    """
    # Normalize H
    H = H / H[2,2]
    K_inv = np.linalg.inv(K)
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    # Compute lambda (scale factor)
    lam = 1.0 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = lam * np.dot(K_inv, h1)
    r2 = lam * np.dot(K_inv, h2)
    t  = lam * np.dot(K_inv, h3)
    # Orthogonalize rotation
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    # Ensure R is a proper rotation matrix using SVD
    U, _, Vt = np.linalg.svd(R)
    R = np.dot(U, Vt)
    return R, t

def compute_all_extrinsics(K, H_all):
    """
    Computes extrinsic parameters for all views.

    Args:
        K (np.ndarray): 3x3 camera intrinsic matrix.
        H_all (list of np.ndarray): List of 3x3 homographies.

    Returns:
        extrinsics (list of (R, t)): One (R, t) pair per image.
    """
    extrinsics = []
    for H in H_all:
        R, t = compute_extrinsic(K, H)
        extrinsics.append((R, t))
    return extrinsics
