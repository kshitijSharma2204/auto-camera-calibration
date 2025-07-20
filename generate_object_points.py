import numpy as np

def generate_object_points(checkerboard_size, square_size=1.0):
  """
  Generates the 3D real-world coordinates of checkerboard corners on the Z=0 plane.
  
  Args:
      checkerboard_size (tuple): (columns, rows) of inner corners (e.g., (9, 6))
      square_size (float): Size of a checkerboard square (any unit, default=1.0)

  Returns:
      objp (np.ndarray): (N, 3) array of 3D points (x, y, 0) for each corner.
  """
  cols, rows = checkerboard_size
  objp = np.zeros((cols * rows, 3), np.float32)
  objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
  objp *= square_size
  return objp

def pair_object_image_points(img_corners_list, checkerboard_size, square_size=1.0):
  """
  Pairs the object points with each set of detected image corners.

  Args:
      img_corners_list (list of np.ndarray): Each element is (N, 2) array of detected corners for one image.
      checkerboard_size (tuple): (columns, rows) of inner corners.
      square_size (float): Physical size of checker square.

  Returns:
      objpoints (list of np.ndarray): List of object points arrays (one per image).
      imgpoints (list of np.ndarray): List of image corners arrays (one per image, filtered for success).
  """
  objp = generate_object_points(checkerboard_size, square_size)
  objpoints = []
  imgpoints = []
  for corners in img_corners_list:
    if corners is not None and len(corners) == objp.shape[0]:
      objpoints.append(objp)
      imgpoints.append(corners)
  return objpoints, imgpoints