import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

cam = 'mynteye'
images_left = glob.glob(cam + '/left*.png')
images_right = glob.glob(cam + '/right*.png')

assert images_right, images_left
images_right.sort()
images_left.sort()
#print(images)

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
        cv2.imshow('img',img)
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
        cv2.imshow('img',img)
        cv2.waitKey(5)
cv2.destroyAllWindows()

ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints_l, imgpoints_l, gray.shape[::-1], None, None, flags = cv2.CALIB_RATIONAL_MODEL)

img_l = cv2.imread(cam + '/left-0001.png')
h,  w = img_l.shape[:2]
newcameramtx_l, roi_l = cv2.getOptimalNewCameraMatrix(mtx_l,dist_l,(w,h),1,(w,h))

ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints_r, imgpoints_r, gray.shape[::-1], None, None, flags = cv2.CALIB_RATIONAL_MODEL)
img_r = cv2.imread(cam + '/right-0001.png')
h,  w = img_r.shape[:2]
newcameramtx_r, roi_r = cv2.getOptimalNewCameraMatrix(mtx_r,dist_r,(w,h),1,(w,h))

# undistort
img_l_undist = cv2.undistort(img_l, mtx_l, dist_l, None, newcameramtx_l)
img_l = cv2.imread(cam + '/left-0001.png')

# cv2.imshow('distorted image', img_l)
# cv2.waitKey()

cv2.imshow('undistorted image', img_l_undist)
cv2.waitKey()
cv2.destroyAllWindows()
