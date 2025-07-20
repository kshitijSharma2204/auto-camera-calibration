import glob
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from detect_checkerboard_corners import getCheckerBoardCorners
from generate_object_points import pair_object_image_points
from estimate_homographies import compute_all_homographies
from compute_intrinsics import compute_intrinsics
from compute_extrinsics import compute_all_extrinsics
from bundle_adjustment import run_bundle_adjustment
from compute_mean_reprojection_error import project_points, compute_mean_reprojection_error
from visualize_and_save import visualizeAndSave

def main():
  # ---- Step 1: Detect corners in all images ----
  checkerboard_size = (9, 6)
  img_dir = "Calibration_Imgs"
  output_dir = "Output"

  image_paths = glob.glob(os.path.join(img_dir, "*.jpg"))
  img_corners_list = []

  for img_path in image_paths:
    img, corners = getCheckerBoardCorners(img_path, checkerboard_size)
    img_corners_list.append(corners)

    img_filename = os.path.basename(img_path)
    img_name, img_ext = os.path.splitext(img_filename)
    output_path = os.path.join(output_dir, f"{img_name}_corners{img_ext}")
    
    visualizeAndSave(img, False, output_path)

  # ---- Step 2: Generate object points and pair with image points ----
  objpoints, imgpoints = pair_object_image_points(img_corners_list, checkerboard_size)

  # ---- Step 3: Estimate homographies ----
  H_all = compute_all_homographies(objpoints, imgpoints)

  # ---- Step 4: Compute intrinsics ----
  K = compute_intrinsics(H_all)
  print("Initial camera intrinsics:\n", K)

  # ---- Step 5: Compute extrinsics ----
  extrinsics_init = compute_all_extrinsics(K, H_all)
  print("Initial extrinsics for first image (R, t):\n", extrinsics_init[0])

  # ---- Step 6: Initialize distortion ----
  dist_init = np.zeros(2)  # [k1, k2] as per Zhang

  # ---- Step 7: Bundle adjustment ----
  K_opt, dist_opt, extrinsics_opt = run_bundle_adjustment(objpoints, imgpoints, K, dist_init, extrinsics_init)
  print("\nOptimized camera intrinsics (K):\n", K_opt)
  print("Optimized distortion coefficients:", dist_opt)
  print("Optimized extrinsics for first image (R, t):\n", extrinsics_opt[0])

  mean_err = compute_mean_reprojection_error(objpoints, imgpoints, K_opt, dist_opt, extrinsics_opt)
  print(f"\nMean reprojection error: {mean_err:.4f} pixels")

  # ---- OpenCV built-in calibration for comparison ----
  print("\n--- OpenCV Calibration (2 radial, 0 tangential) ---")
  sample_img = cv2.imread(image_paths[0], 0)  # 0 = grayscale
  img_shape = sample_img.shape[::-1]  # (width, height)
  flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
  ret, K_cv, dist_cv, rvecs_cv, tvecs_cv = cv2.calibrateCamera(
      objpoints, imgpoints, img_shape, None, None, flags=flags)
  print("OpenCV camera matrix:\n", K_cv)
  print("OpenCV distortion coefficients:\n", dist_cv.ravel())
  print("OpenCV mean reprojection error: %.4f pixels" % (ret))


  # ---- Visualize detected vs. reprojected points ----
  print("\nVisualizing detected (green) vs reprojected (red) points for the first image...")
  img = cv2.imread(image_paths[0])
  R, t = extrinsics_opt[0]
  objp = objpoints[0]
  img_proj = project_points(objp, K_opt, dist_opt, R, t)
  for (u, v) in imgpoints[0]:
      cv2.circle(img, (int(u), int(v)), 6, (0, 255, 0), 2) # Green = detected
  for (u, v) in img_proj:
      cv2.circle(img, (int(u), int(v)), 3, (0, 0, 255), -1) # Red = projected
  
  # Save the comparison image
  comparison_output_path = os.path.join(output_dir, "detected_vs_reprojected_points.jpg")
  visualizeAndSave(img, False, comparison_output_path)
  
  plt.figure(figsize=(8, 6))
  plt.title("First Image: Detected (green) vs. Reprojected (red) Points")
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  plt.axis('off')
  plt.show()

  # ---- Visualize undistortion ----
  print("\nShowing original vs. undistorted first image...")
  img_original = cv2.imread(image_paths[0])
  dist_opt_padded = np.array([dist_opt[0], dist_opt[1], 0.0, 0.0, 0.0])
  img_undist = cv2.undistort(img_original, K_opt, dist_opt_padded)
  
  # Save original and undistorted images separately
  original_output_path = os.path.join(output_dir, "original_image.jpg")
  undistorted_output_path = os.path.join(output_dir, "undistorted_image.jpg")
  visualizeAndSave(img_original, False, original_output_path)
  visualizeAndSave(img_undist, False, undistorted_output_path)
  
  # Create side-by-side comparison
  h, w = img_original.shape[:2]
  comparison_img = np.zeros((h, w*2, 3), dtype=np.uint8)
  comparison_img[:, :w] = img_original
  comparison_img[:, w:] = img_undist
  
  # Add text labels to the comparison image
  font = cv2.FONT_HERSHEY_SIMPLEX
  cv2.putText(comparison_img, 'Original', (50, 50), font, 2, (255, 255, 255), 3)
  cv2.putText(comparison_img, 'Undistorted', (w + 50, 50), font, 2, (255, 255, 255), 3)
  
  # Save the side-by-side comparison
  comparison_undist_path = os.path.join(output_dir, "original_vs_undistorted_comparison.jpg")
  visualizeAndSave(comparison_img, False, comparison_undist_path)
  
  plt.figure(figsize=(12,6))
  plt.subplot(1,2,1)
  plt.title('Original')
  plt.imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
  plt.axis('off')
  plt.subplot(1,2,2)
  plt.title('Undistorted')
  plt.imshow(cv2.cvtColor(img_undist, cv2.COLOR_BGR2RGB))
  plt.axis('off')
  plt.show()

if __name__ == "__main__":
  main()
