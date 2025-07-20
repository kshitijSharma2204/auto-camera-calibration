import numpy as np
from bundle_adjustment import project_points

def compute_mean_reprojection_error(objpoints, imgpoints, K, dist, extrinsics):
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        objp = objpoints[i]
        imgp = imgpoints[i]
        R, t = extrinsics[i]

        # Project 3D object points to 2D image points
        imgp_proj = project_points(objp, K, dist, R, t)

        # Compute error (Euclidean distance per point)
        error = np.linalg.norm(imgp - imgp_proj, axis=1).sum()
        total_error += error
        total_points += len(objp)

    mean_error = total_error / total_points
    return mean_error