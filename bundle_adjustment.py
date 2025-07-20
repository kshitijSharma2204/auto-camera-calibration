import numpy as np
import cv2
from scipy.optimize import least_squares

def project_points(objpoints, K, dist_coeffs, R, t):
    """
    Projects 3D object points onto the image plane using K, distortion, R, t.
    Args:
        objpoints: (N, 3) 3D world points
        K: (3, 3) intrinsics
        dist_coeffs: (2,) [k1, k2] radial distortion
        R: (3, 3) rotation
        t: (3,) translation
    Returns:
        imgpoints_proj: (N, 2) projected 2D points
    """
    # Transform points to camera coords
    obj_cam = (R @ objpoints.T + t.reshape(3,1)).T  # shape (N, 3)
    x = obj_cam[:, 0] / obj_cam[:, 2]
    y = obj_cam[:, 1] / obj_cam[:, 2]
    r2 = x**2 + y**2
    # Apply radial distortion
    x_dist = x * (1 + dist_coeffs[0]*r2 + dist_coeffs[1]*r2**2)
    y_dist = y * (1 + dist_coeffs[0]*r2 + dist_coeffs[1]*r2**2)
    # Project with K
    u = K[0,0]*x_dist + K[0,1]*y_dist + K[0,2]
    v = K[1,1]*y_dist + K[1,2]
    return np.vstack([u, v]).T  # shape (N, 2)

def pack_params(K, dist_coeffs, extrinsics):
    """
    Packs parameters for optimization.
    """
    params = []
    # Intrinsics: fx, skew, cx, fy, cy
    params += [K[0,0], K[0,1], K[0,2], K[1,1], K[1,2]]
    # Distortion: k1, k2
    params += list(dist_coeffs)
    # Extrinsics for each image: (rvec, t)
    for (R, t) in extrinsics:
        rvec, _ = cv2.Rodrigues(R)
        params += rvec.flatten().tolist() + t.flatten().tolist()
    return np.array(params, dtype=np.float64)

def unpack_params(params, n_images):
    """
    Unpacks parameter vector.
    """
    fx, skew, cx, fy, cy = params[0:5]
    dist_coeffs = params[5:7]
    K = np.array([[fx, skew, cx], [0, fy, cy], [0,0,1]])
    extrinsics = []
    idx = 7
    for _ in range(n_images):
        rvec = np.array(params[idx:idx+3])
        t = np.array(params[idx+3:idx+6])
        R, _ = cv2.Rodrigues(rvec)
        extrinsics.append((R, t))
        idx += 6
    return K, dist_coeffs, extrinsics

def reprojection_residuals(params, objpoints_list, imgpoints_list):
    """
    Computes residuals between observed and projected points for all images.
    """
    n_images = len(objpoints_list)
    K, dist_coeffs, extrinsics = unpack_params(params, n_images)
    residuals = []
    for i in range(n_images):
        objp = objpoints_list[i]
        imgp = imgpoints_list[i]
        R, t = extrinsics[i]
        imgp_proj = project_points(objp, K, dist_coeffs, R, t)
        residuals.extend((imgp_proj - imgp).ravel())
    return np.array(residuals)

def run_bundle_adjustment(objpoints_list, imgpoints_list, K_init, dist_init, extrinsics_init):
    """
    Main entry: optimizes all parameters.
    """
    params0 = pack_params(K_init, dist_init, extrinsics_init)
    result = least_squares(
        reprojection_residuals, params0,
        args=(objpoints_list, imgpoints_list),
        verbose=2, jac='2-point', method='lm'
    )
    K_opt, dist_opt, extrinsics_opt = unpack_params(result.x, len(objpoints_list))
    return K_opt, dist_opt, extrinsics_opt