import cv2

def visualizeAndSave(img, display=False, output_path=None):
  """
  Visualizes and/or saves an image.

  Args:
    img: The image to process (numpy array).
    display: If True, display the image in a window.
    output_path: If provided, save the image to this path.

  Returns:
    The processed image.
  """
  if display:
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  if output_path is not None:
    cv2.imwrite(output_path, img)
  return img
