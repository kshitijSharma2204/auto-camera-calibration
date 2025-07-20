import cv2
import numpy as np

def getCheckerBoardCorners(
    image_path,
    checkerboard_size,
    circle_radius=8,
    circle_color=(0, 0, 255),
    circle_thickness=10,   # -1 means filled
    draw_lines=True,
    line_color=(0, 255, 0),
    line_thickness=4
):
    """
    Detects checkerboard corners and makes them more pronounced on the image.

    Args:
        image_path (str): Path to the image.
        checkerboard_size (tuple): (columns, rows) number of inner corners (e.g., (9, 6))
        circle_radius (int): Radius of the drawn circles.
        circle_color (tuple): Color for the circles (B, G, R).
        circle_thickness (int): Thickness of the circle outline. Use -1 for filled.
        draw_lines (bool): Whether to draw lines connecting the corners.
        line_color (tuple): Color for the connecting lines (B, G, R).
        line_thickness (int): Thickness of the connecting lines.

    Returns:
        img (np.ndarray or None): Image with pronounced corners drawn.
        corners (np.ndarray or None): Detected corners, shape (N, 2), or None if not found.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        corners2d = corners.squeeze()

        # Draw extra circles to highlight corners
        for pt in corners2d:
            center = tuple(np.round(pt).astype(int))
            cv2.circle(img, center, circle_radius, circle_color, circle_thickness)

        # Draw thick lines connecting the corners (row-wise and column-wise)
        if draw_lines:
            cols, rows = checkerboard_size
            # Draw horizontal lines
            for r in range(rows):
                for c in range(cols - 1):
                    idx1 = r * cols + c
                    idx2 = r * cols + c + 1
                    pt1 = tuple(np.round(corners2d[idx1]).astype(int))
                    pt2 = tuple(np.round(corners2d[idx2]).astype(int))
                    cv2.line(img, pt1, pt2, line_color, line_thickness)
            # Draw vertical lines
            for c in range(cols):
                for r in range(rows - 1):
                    idx1 = r * cols + c
                    idx2 = (r + 1) * cols + c
                    pt1 = tuple(np.round(corners2d[idx1]).astype(int))
                    pt2 = tuple(np.round(corners2d[idx2]).astype(int))
                    cv2.line(img, pt1, pt2, line_color, line_thickness)

        return img, corners2d
    else:
        return None, None
