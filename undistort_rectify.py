# %% IMPORTS
import numpy as np
import cv2
import glob

# %% LOAD IMAGES
dir_dataset = '../datasets/conveyor_without_occlusions'
images_left = glob.glob(dir_dataset + '/left/*.png')
images_right = glob.glob(dir_dataset + '/right/*.png')

assert images_right, images_left
assert (len(images_right) == len(images_left))
n_images = len(images_right)
images_right.sort()
images_left.sort()

# %% LOAD RECTIFIED PROJECTION MATRICES AND UNDISTORTION/RECTIFICATION MAPS
dir_params = "../calibration/calibration_matrix/"
mtx_P_l = np.load(dir_params + "projection_matrix_l.npy")
mtx_P_r = np.load(dir_params + "projection_matrix_r.npy")
rect_map_l_x = np.load(dir_params + "map_l_x.npy")
rect_map_l_y = np.load(dir_params + "map_l_y.npy")
rect_map_r_x = np.load(dir_params + "map_r_x.npy")
rect_map_r_y = np.load(dir_params + "map_r_y.npy")

# %% SETTINGS
h, w = cv2.imread(images_left[0]).shape[:2] # size of the images (pixels)

# %% REAL TIME RECTIFICATION

# for each frame of the video:
for i in range(n_images):
	# grab current frame (left and right)
	frame_rgb_curr = cv2.imread(images_left[i])
	frame_rgb_curr_right = cv2.imread(images_right[i])
	
	# undistort and rectify left and right image
	frame_rgb_curr_undist_rect = cv2.remap(frame_rgb_curr, rect_map_l_x, rect_map_l_y, cv2.INTER_LINEAR)
	frame_rgb_curr_undist_rect_right = cv2.remap(frame_rgb_curr_right, rect_map_r_x, rect_map_r_y, cv2.INTER_LINEAR)
	
	# display final result
	cv2.putText(frame_rgb_curr_undist_rect, "Undistorted rectified (Left)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.putText(frame_rgb_curr_undist_rect_right, "Undistorted rectified (Right)", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	img_l_r = np.hstack((frame_rgb_curr_undist_rect,frame_rgb_curr_undist_rect_right))
	windowname_1 = "Undistorted and rectified"
	cv2.namedWindow(windowname_1, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(windowname_1, (1500,400))
	cv2.imshow(windowname_1,img_l_r)

	if (cv2.waitKey(1) & 0xFF == 27): break  # ESC to quit
	
	

cv2.destroyAllWindows()