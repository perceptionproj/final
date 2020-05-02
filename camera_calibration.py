# %% IMPORTS
import numpy as np
import cv2
import glob

# %% LOAD IMAGES
cam = '../calibration/calibration_patterns'
images_left = glob.glob(cam + '/left/*.png')
images_right = glob.glob(cam + '/right/*.png')

assert images_right, images_left
assert (len(images_right) == len(images_left))
n_images = len(images_right)
images_right.sort()
images_left.sort()

# %% INITIALIZE CHECKERBOARD OBJECT POINTS
nb_vertical = 9
nb_horizontal = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # vector of vectors of calibration pattern points in the calibration pattern coordinate space
imgpoints_l = [] # 2d points in image plane.
imgpoints_r = [] # 2d points in image plane.

# %% EXTRACT CHECKERBOARD CORNERS (AND DISPLAY THEM)
for i in range(n_images):
    img_l = cv2.imread(images_left[i])
    img_r = cv2.imread(images_right[i])
    gray_l = cv2.cvtColor(img_l,cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r,cv2.COLOR_BGR2GRAY)
    ret_l, corners_l = cv2.findChessboardCorners(gray_l, patternSize = (nb_vertical, nb_horizontal))
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, patternSize = (nb_vertical, nb_horizontal))
	
    # If found, add object points, image points (after refining them)
    if ret_l == True and ret_r == True:
        objpoints.append(objp)
        imgpoints_l.append(corners_l)
        imgpoints_r.append(corners_r)
        # Draw and display the corners
        img_l = cv2.drawChessboardCorners(img_l, (nb_vertical,nb_horizontal), corners_l,ret_l)
        img_r = cv2.drawChessboardCorners(img_r, (nb_vertical,nb_horizontal), corners_r,ret_r)
        cv2.putText(img_l, "Left Camera", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(img_r, "Right Camera", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        img_l_r = np.vstack((img_l,img_r))
        windowname_1 = "Calibration Pattern (Left and Right Camera)"
        cv2.namedWindow(windowname_1, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowname_1, 600, 600)
        cv2.imshow(windowname_1,img_l_r)
        cv2.waitKey(5)
cv2.destroyAllWindows()

# %% STEREOCALIBRATE - COMPUTE (INTRINSIC) CAMERA MATRICES, DISTORTION COEFFICIENTS, ROTATION AND TRANSLATION BETWEEN THE TWO CAMERAS AND REPROJECTION ERROR
assert (img_l.shape[:2] == img_r.shape[:2])
h, w = img_l.shape[:2]

term_crit_sc = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
flags_sc = cv2.CALIB_RATIONAL_MODEL
ret_stereo,  mtx_l, dist_l, mtx_r, dist_r, mtx_R, mtx_T, mtx_E, mtx_F = cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, None, None, None, None, (w,h), flags=flags_sc, criteria=term_crit_sc)


# %% STEREORECTIFY - COMPUTE AND SAVE THE RECTIFICATION TRANSFORM AND PROJECTION MATRIX OF THE 2 CAMERAS, USING THE MATRICES COMPUTED BY STEREOCALIBRATE

mtx_R_l, mtx_R_r, mtx_P_l, mtx_P_r, mtx_Q, roi_rec_l, roi_rec_r = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (w,h), mtx_R, mtx_T, alpha=0)

# %% COMPUTE UNDISTORTION AND RECTIFICATION TRANSFORMATION MAP
map1x, map1y = cv2.initUndistortRectifyMap(mtx_l,dist_l,mtx_R_l,mtx_P_l,(w,h),cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(mtx_r,dist_r,mtx_R_r,mtx_P_r,(w,h),cv2.CV_32FC1)


# %% SAVE RECTIFIED CAMERA MATRICES AND UNDISTORTION/RECTIFICATION MAPS FOR FUTURE USE
dir_calib = "../calibration/calibration_matrix/"
np.save(dir_calib + "projection_matrix_l", mtx_P_l)
np.save(dir_calib + "projection_matrix_r", mtx_P_r)
np.save(dir_calib + "map_l_x", map1x)
np.save(dir_calib + "map_l_y", map1y)
np.save(dir_calib + "map_r_x", map2x)
np.save(dir_calib + "map_r_y", map2y)


# %% CHECK RESULT CORRECTNESS

img_l = cv2.imread(images_left[0])
cv2.imshow('Distorted Image', img_l)
cv2.waitKey()

img_l_undist_rect = cv2.remap(img_l, map1x, map1y, cv2.INTER_LINEAR)
cv2.imshow('Undistorted and Rectified Image', img_l_undist_rect)
cv2.waitKey()

cv2.destroyAllWindows()