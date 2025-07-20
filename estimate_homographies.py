import numpy as np

def compute_homography(objp, imgp):
    """
    Estimates the homography H that maps object points to image points.

    Args:
        objp (np.ndarray): (N, 3) array of object points (x, y, 0).
        imgp (np.ndarray): (N, 2) array of detected image points (u, v).

    Returns:
        H (np.ndarray): 3x3 homography matrix.
    """
    assert objp.shape[0] == imgp.shape[0], "Number of object and image points must match."
    N = objp.shape[0]
    A = []
    for i in range(N):
        X, Y = objp[i, 0], objp[i, 1]
        x, y = imgp[i, 0], imgp[i, 1]
        A.append([-X, -Y, -1,  0,  0,  0, X*x, Y*x, x])
        A.append([ 0,  0,  0, -X, -Y, -1, X*y, Y*y, y])
    A = np.array(A)
    # Solve Ah = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    H = H / H[2,2]
    return H

def compute_all_homographies(objpoints, imgpoints):
    """
    Computes homographies for all image/object point sets.

    Args:
        objpoints (list of np.ndarray): List of object points arrays.
        imgpoints (list of np.ndarray): List of image points arrays.

    Returns:
        H_all (list of np.ndarray): List of 3x3 homography matrices.
    """
    H_all = []
    for objp, imgp in zip(objpoints, imgpoints):
        H = compute_homography(objp, imgp)
        H_all.append(H)
    return H_all