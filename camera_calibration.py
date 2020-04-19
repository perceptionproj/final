# %% IMPORTS
import numpy as np
import cv2
import glob

# %% LOAD IMAGES
cam = '../calibration/calibration_patterns'
images_left = glob.glob(cam + '/left/*.png')
images_right = glob.glob(cam + '/right/*.png')

assert images_right, images_left
images_right.sort()
images_left.sort()

# %% INITIALIZE OBJECT POINTS
nb_vertical = 9
nb_horizontal = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints_l = [] # 3d point in real world space
imgpoints_l = [] # 2d points in image plane.

objpoints_r = [] # 3d point in real world space
imgpoints_r = [] # 2d points in image plane.

# %% EXTRACT CHECKERBOARD CORNERS (AND DISPLAY THEM)
for fname in images_left:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, patternSize = (nb_vertical, nb_horizontal))
	
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints_l.append(objp)
        imgpoints_l.append(corners)
		
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nb_vertical,nb_horizontal), corners,ret)
        cv2.imshow('Left Calibration Pattern',img)
        cv2.waitKey(5)
cv2.destroyAllWindows()


for fname in images_right:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, patternSize = (nb_vertical, nb_horizontal))
	
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints_r.append(objp)
        imgpoints_r.append(corners)
		
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (nb_vertical,nb_horizontal), corners,ret)
        cv2.imshow('Right Calibration Pattern',img)
        cv2.waitKey(5)
cv2.destroyAllWindows()

# %% CALIBRATE CAMERA (AND SAVE DISTORTION COEFFICIENTS)
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints_l, imgpoints_l, gray.shape[::-1], None, None, flags = cv2.CALIB_RATIONAL_MODEL)
img_l = cv2.imread(images_left[0])
h,  w = img_l.shape[:2]
newcameramtx_l, roi_l = cv2.getOptimalNewCameraMatrix(mtx_l,dist_l,(w,h),1,(w,h))

ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints_r, imgpoints_r, gray.shape[::-1], None, None, flags = cv2.CALIB_RATIONAL_MODEL)
img_r = cv2.imread(images_right[0])
h,  w = img_r.shape[:2]
newcameramtx_r, roi_r = cv2.getOptimalNewCameraMatrix(mtx_r,dist_r,(w,h),1,(w,h))

# Saving coefficients
dir = "../calibration/calibration_matrix/"
np.save(dir + "mtx_l", mtx_l)
np.save(dir + "dist_l", dist_l)
np.save(dir + "mtx_r", mtx_r)
np.save(dir + "dist_r", dist_r)

# %% UNDISTORT, CROP AND SHOW IMAGE
# undistort
img_l_undist = cv2.undistort(img_l, mtx_l, dist_l, None, newcameramtx_l)
img_l = cv2.imread(images_left[0])

cv2.imshow('Distorted Image', img_l)
cv2.waitKey()
cv2.imshow('Undistorted Image', img_l_undist)
cv2.waitKey()

# crop the image
x,y,w,h = roi_l
img_l_undist_crop = img_l_undist[y:y+h, x:x+w]

cv2.imshow('Undistorted and Cropped Image', img_l_undist_crop)
cv2.waitKey()

cv2.destroyAllWindows()