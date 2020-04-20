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

# %% INITIALIZE OBJECT POINTS
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
        cv2.namedWindow("Calibration Pattern (Left and Right Cameras)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Calibration Pattern (Left and Right Cameras)", 600, 600)
        cv2.imshow("Calibration Pattern (Left and Right Cameras)",img_l_r)
        cv2.waitKey(5)
cv2.destroyAllWindows()

# %% CALIBRATECAMERA - COMPUTE AND SAVE (INTRINSIC) CAMERA MATRICES AND DISTORTION COEFFICIENTS OF EACH CAMERA INDIVIDUALLY
assert (img_l.shape[:2] == img_r.shape[:2])
h,  w = img_l.shape[:2]

ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, imgpoints_l, (w,h), None, None, flags = cv2.CALIB_RATIONAL_MODEL)
mtx_l_new, roi_cal_l = cv2.getOptimalNewCameraMatrix(mtx_l,dist_l,(w,h),1,(w,h))

ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, imgpoints_r, (w,h), None, None, flags = cv2.CALIB_RATIONAL_MODEL)
mtx_r_new, roi_cal_r = cv2.getOptimalNewCameraMatrix(mtx_r,dist_r,(w,h),1,(w,h))

# Saving camera matrices and distortion coefficients
dir = "../calibration/calibration_matrix/"
np.save(dir + "camera_matrix_l", mtx_l)
np.save(dir + "distortion_coeff_l", dist_l)
np.save(dir + "camera_matrix_l_new", mtx_l_new)
np.save(dir + "camera_matrix_r", mtx_r)
np.save(dir + "distortion_coeff_r", dist_r)
np.save(dir + "camera_matrix_r_new", mtx_r_new)

# %% STEREOCALIBRATE - COMPUTE AND SAVE ROTATION AND TRANSLATION BETWEEN THE TWO CAMERAS, USING THE DISTORTION COEFFICIENTS AND CAMERA MATRICES PROVIDED BY CALIBRATECAMERA
term_crit = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
ret_stereo, _, _, _, _, mtx_R, mtx_T, mtx_E, mtx_F = cv2.stereoCalibrate(objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, (w,h), flags=cv2.CALIB_FIX_INTRINSIC, criteria=term_crit)

np.save(dir + "rotation_l_r", mtx_R)
np.save(dir + "translation_l_r", mtx_T)
np.save(dir + "essential_matrix", mtx_E)
np.save(dir + "fundamental_matrix", mtx_F)

# %% STEREORECTIFY - COMPUTE AND SAVE THE RECTIFICATION TRANSFORM AND PROJECTION MATRIX OF THE 2 CAMERAS, USING THE MATRICES COMPUTED BY STEREOCALIBRATE
mtx_R_l, mtx_R_r, mtx_P_l, mtx_P_r, mtx_Q, roi_rec_l, roi_rec_r = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (w,h), mtx_R, mtx_T) 

np.save(dir + "projection_matrix_l", mtx_P_l)
np.save(dir + "projection_matrix_r", mtx_P_r)

# %% UNDISTORT, CROP AND SHOW IMAGE
# undistort
img_l = cv2.imread(images_left[0])
img_l_undist = cv2.undistort(img_l, mtx_l, dist_l, None, mtx_l_new)

cv2.imshow('Distorted Image', img_l)
cv2.waitKey()
cv2.imshow('Undistorted Image', img_l_undist)
cv2.waitKey()

# crop the image
x,y,w,h = roi_cal_l
img_l_undist_crop = img_l_undist[y:y+h, x:x+w]

cv2.imshow('Undistorted and Cropped Image', img_l_undist_crop)
cv2.waitKey()

cv2.destroyAllWindows()