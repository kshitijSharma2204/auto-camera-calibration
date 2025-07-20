import cv2

def getCheckerBoardCorners(image_path, checkerboard_size):
  """
  Detects checkerboard corners in an image.

  Args:
      image_path (str): Path to the image.
      checkerboard_size (tuple): (columns, rows) number of inner corners (e.g., (9, 6))
      display (bool): If True, display the image with drawn corners.
      save_path (str or None): If given, saves the result image to this path.

  Returns:
      corners (np.ndarray or None): Detected corners, shape (N, 2), or None if not found.
  """
  img = cv2.imread(image_path)
  
  if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

  if ret:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
    return img, corners.squeeze()
  else:
    return None
